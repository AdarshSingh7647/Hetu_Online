"""
Model-specific prompt construction for eval, mirroring the exact same
model-card rules used at training time: DeepSeek-distill models get no
system prompt and put everything in the user turn, plus a forced
`<think>\\n` response opener; Qwen3 uses its native enable_thinking chat
template.

Critically, the CotCond/CotGen distinction is NEVER reflected in the
prompt -- both modes see byte-identical prompts for a given
(model_family, benchmark, question). Only the checkpoint/adapter loaded
differs. This module has no knowledge of which mode is being evaluated.

NOTE: this is a DIFFERENT, wider family vocabulary than
hetu_online.model_families (the training-side registry, currently
qwen3/gemma4 only) -- eval has always covered qwen3 sizes plus two
DeepSeek-R1-Distill families that were never part of training's registry
here (those checkpoints were trained via a different, non-Hetu_Online
pipeline and are evaluated here for comparison only). Do not assume the
two modules' model_family strings are interchangeable.
"""

MODEL_FAMILIES = ("qwen3", "dsr1_llama8b", "dsr1_qwen7b", "qwen3_0p6b", "qwen3_4b")

_FAMILY_TO_TEMPLATE = {
    "qwen3": "qwen3",
    "dsr1_llama8b": "deepseekr1",
    "dsr1_qwen7b": "deepseekr1",
    "qwen3_0p6b": "qwen3",
    "qwen3_4b": "qwen3",
}

# Bare HF repo ids -- transformers/vLLM resolve and cache these under
# HF_HOME on whatever machine this runs on.
_FAMILY_TO_BASE_MODEL = {
    "qwen3": "Qwen/Qwen3-8B",
    "dsr1_llama8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "dsr1_qwen7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # Same architecture/tokenizer/chat-template family as "qwen3" (Qwen3-8B)
    # -- just a different checkpoint size. Everything else (template,
    # thinking flag, sampling params, context window, continuation-round
    # budget) is identical, so these two reuse every qwen3-keyed value below
    # rather than duplicating values that would only ever match "qwen3"'s.
    "qwen3_0p6b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B",
}


def base_model_path(model_family: str) -> str:
    return _FAMILY_TO_BASE_MODEL[model_family]


def build_chat_messages(model_family: str, question: str) -> list[dict]:
    """Returns the chat-format message list to pass to the tokenizer's
    apply_chat_template / vLLM's chat entrypoint. No system message for
    DeepSeek-distill families (model-card rule); Qwen3 gets none either
    here since the benchmark question already stands alone."""
    return [{"role": "user", "content": question}]


def forced_assistant_prefix(model_family: str) -> str:
    """Text already present at the end of the rendered prompt that opens
    the assistant turn's reasoning block -- needed so split_think_answer()
    can reconstruct a complete <think>...</think> span from prompt+generated
    text. Both families' raw chat templates already auto-append this at
    add_generation_prompt time (verified directly): DeepSeek-distill's
    template unconditionally ends with "<｜Assistant｜><think>\\n", and
    Qwen3's enable_thinking=True leaves the prompt open at
    "<|im_start|>assistant\\n" with the model expected to open its own
    <think> block. So DeepSeek's case returns the tag the template already
    emitted (do NOT re-append it -- that would duplicate the tag), and
    Qwen3 returns "" since nothing is pre-opened for it."""
    if model_family in ("dsr1_llama8b", "dsr1_qwen7b"):
        return "<think>\n"
    return ""


def needs_enable_thinking(model_family: str) -> bool:
    return model_family in ("qwen3", "qwen3_0p6b", "qwen3_4b")


# Sampling params per model card (both confirmed directly from each model's
# HF card, not assumed):
#   - Qwen3-8B (thinking mode): temperature=0.6, top_p=0.95, top_k=20,
#     min_p=0; card explicitly warns against greedy decoding (causes
#     performance degradation/endless loops) and recommends an output
#     length of 32768 tokens for most queries (38912 for hard math/code).
#   - DeepSeek-R1-Distill-{Llama-8B,Qwen-7B}: temperature=0.6, top_p=0.95
#     (card's own eval protocol: pass@1 estimated over 64 samples at these
#     settings), also warns against greedy decoding, recommends max
#     generation length of 32768 tokens. No top_k/min_p mentioned in the
#     card -- left at vLLM's defaults (top_k=-1/disabled, min_p=0) rather
#     than guessed.
SAMPLING_PARAMS_BY_FAMILY = {
    "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "dsr1_llama8b": {"temperature": 0.6, "top_p": 0.95},
    "dsr1_qwen7b": {"temperature": 0.6, "top_p": 0.95},
    # Qwen3-0.6B/4B share Qwen3-8B's card-recommended thinking-mode sampling
    # params (same model family/card, just a different parameter count).
    "qwen3_0p6b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "qwen3_4b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
}

# Card-recommended max generation length, per family (used as the
# eval harness's default --max_new_tokens unless overridden per-benchmark).
RECOMMENDED_MAX_NEW_TOKENS = {
    "qwen3": 32768,
    "dsr1_llama8b": 32768,
    "dsr1_qwen7b": 32768,
    "qwen3_0p6b": 32768,
    "qwen3_4b": 32768,
}

# Each model's OWN native context window (max_position_embeddings, confirmed
# directly from each model's config.json, not assumed): Qwen3-8B is capped
# at 40960, while both DeepSeek-R1-Distill models support 131072. This
# matters for the continuation-round design (see run_eval.py's
# EvalConfig.required_max_model_len): naively wanting 32768 max_new_tokens x
# (1 initial + 2 continuation rounds) + prompt headroom needs ~102k tokens
# of context, which fits DeepSeek fine but is IMPOSSIBLE for Qwen3 (real
# failure hit in production: vLLM rejected a continuation request with
# "decoder prompt (length 36864) plus requested output tokens is longer than
# the maximum model length of 36864" once max_model_len was sized for only
# one round). Qwen3 gets fewer continuation rounds instead of silently
# failing or (worse) attempting context-length tricks (e.g. RoPE scaling)
# that were never validated for this model.
NATIVE_MAX_CONTEXT = {
    "qwen3": 40960,
    "dsr1_llama8b": 131072,
    "dsr1_qwen7b": 131072,
    # Both confirmed directly from each checkpoint's own config.json
    # (max_position_embeddings), not assumed from the 8B's value: Qwen3-0.6B
    # and Qwen3-4B both also cap at 40960.
    "qwen3_0p6b": 40960,
    "qwen3_4b": 40960,
}

# Continuation rounds per family, chosen so
# prompt_headroom + max_new_tokens + max_continuation_rounds * continuation_max_new_tokens
# stays comfortably under NATIVE_MAX_CONTEXT at the card-recommended 32768
# tokens/round: Qwen3 gets 1 round (4096 + 32768*2 = 69632 -- still over its
# 40960 ceiling at the full 32768/round budget, so Qwen3's continuation round
# additionally uses a smaller per-round budget, see
# CONTINUATION_MAX_NEW_TOKENS_OVERRIDE below); DeepSeek keeps the full 2
# rounds at 32768/round (4096 + 32768*3 = 102400, comfortably under 131072).
MAX_CONTINUATION_ROUNDS_BY_FAMILY = {
    "qwen3": 1,
    "dsr1_llama8b": 2,
    "dsr1_qwen7b": 2,
    # Same 40960 ceiling as Qwen3-8B (verified per-checkpoint, see
    # NATIVE_MAX_CONTEXT), so the same 1-round budget applies.
    "qwen3_0p6b": 1,
    "qwen3_4b": 1,
}

# Qwen3's single continuation round still needs a reduced per-round budget
# to fit: 4096 (headroom) + 32768 (initial) + X <= 40960  =>  X <= 4096.
# Rounded down to a clean 4000 tokens -- enough to let the model wrap up a
# reasoning chain that was cut off near the end, not enough to sustain a
# whole second full reasoning pass (which wouldn't fit in this model's
# context regardless of round count). DeepSeek continuation rounds use the
# same 32768 budget as the initial round (no override needed, has headroom).
CONTINUATION_MAX_NEW_TOKENS_OVERRIDE = {
    "qwen3": 4000,
    "qwen3_0p6b": 4000,
    "qwen3_4b": 4000,
}


def sampling_params_for(model_family: str) -> dict:
    return dict(SAMPLING_PARAMS_BY_FAMILY[model_family])


def max_continuation_rounds_for(model_family: str) -> int:
    return MAX_CONTINUATION_ROUNDS_BY_FAMILY[model_family]


def continuation_max_new_tokens_for(model_family: str, max_new_tokens: int) -> int:
    return CONTINUATION_MAX_NEW_TOKENS_OVERRIDE.get(model_family, max_new_tokens)
