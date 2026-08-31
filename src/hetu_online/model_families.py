"""
Per-model-family config for the two model families HETU_Online supports:
Qwen3 (any size) and Gemma-4 (E2B/E4B/12B). No other family is supported --
a wrong template or think-tag pair here fails SILENTLY (CotCond's masking
fails open to full supervision with no error, see sitecustomize.py's own
docstring), so this module exists to make the validated values the only
reachable ones, rather than free-text flags a caller could typo or
mismatch between build-data and train.

Do not add a new model family here without the same level of verification
these two entries have (cited sources below) -- an unverified addition is
worse than not supporting the model at all, because it fails quietly.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

MODEL_FAMILIES = ("qwen3", "gemma4")


class FamilyConfig(NamedTuple):
    template: str
    think_open: str
    think_close: str
    freeze_vision: bool
    flash_attn: Optional[str]
    enable_liger_kernel: bool
    default_cutoff_len: int


# qwen3: Hetu's original defaults, fully validated on Qwen3-0.6B/4B/8B.
# No multimodal submodules, no known flash-attn/liger dispatch issues.
_QWEN3 = FamilyConfig(
    template="qwen3",
    think_open="<think>\n",
    think_close="\n</think>\n\n",
    freeze_vision=False,
    flash_attn=None,
    enable_liger_kernel=False,
    default_cutoff_len=32768,
)

# gemma4: template is "gemma4", NOT "gemma4n". LLaMA-Factory's own model
# registry (extras/constants.py) maps the E2B/E4B/12B checkpoints to
# "gemma4n" by default, but that default is wrong for text-only training.
# gemma4 and gemma4n are byte-identical in LLaMA-Factory's template
# registry (data/template.py) -- same formatters, same thought_words,
# same ReasoningTemplate -- EXCEPT gemma4n's mm_plugin additionally
# declares audio_token="<|audio|>". That one difference is load-bearing:
# data/collator.py gates its fake-audio injection on
# `template.mm_plugin.audio_token is not None`, so under gemma4n every
# text-only batch got a dummy waveform and ran the audio tower, which
# crashes under flash_attn (ValueError: not enough values to unpack
# (expected 4, got 2)) because transformers' Gemma4 audio tower
# unconditionally 4-way-unpacks a mask that FA2's mask function returns as
# 2D. With gemma4, audio_token is None, no fake audio is injected, the
# audio tower never runs, and the crash cannot occur. Upstream PR #10803's
# own text-only Gemma-4 example likewise uses template: gemma4.
#
# think_open/think_close verified two independent ways: (1) directly from
# the real google/gemma-4-E2B-it chat_template.jinja (HF Hub model API,
# config.chat_template_jinja) -- the thought-wrapping literal is
# '<|channel>thought\n' + thinking_text + '\n<channel|>' (the '\n' before
# the closing literal is the template's own separator, not part of the
# close tag string); (2) from LLaMA-Factory's gemma4/gemma4n
# ReasoningTemplate registration (data/template.py),
# thought_words=("<|channel>thought\n", "<channel|>"). The close tag
# itself has NO leading newline in either source -- do not "symmetrize" it
# to match the open tag's trailing newline, that would not match what
# LLaMA-Factory's template renders and CotCond masking would fail open
# silently.
#
# freeze_vision=True: Gemma-4 E2B/E4B/12B are multimodal checkpoints even
# though this tool only ever trains text-only. Freezing vision_tower +
# multi_modal_projector confines LoRA to language-model linear layers.
# Verified as a safe no-op on 12B (encoder-free architecture -- no such
# submodules exist) via LLaMA-Factory's model_utils/visual.py
# substring-matching design (get_forbidden_modules/find_all_linear_modules
# test substring containment against named_modules(), never getattr/index
# by name, so on a model with no matching submodule the check is just
# always False -- harmless, not a crash).
#
# flash_attn="fa2" + enable_liger_kernel=True: required together at long
# context. vocab_size is 262144 (real config.json), so the un-fused logits
# tensor alone is ~15GB in bf16 at cutoff_len ~28k -- liger's fused linear
# cross-entropy avoids materializing it, and AttentionFunction.AUTO
# (LLaMA-Factory's default) is a no-op for model_type=gemma4 (only
# gemma2/gpt_oss/youtu get special AUTO handling), so flash_attn must be
# requested explicitly or it silently stays on SDPA. NOTE: liger's
# gemma4/gemma4_text dispatch requires HETU_Online's vendored LLaMA-Factory
# patch (patches/llamafactory_gemma4_liger.patch) -- upstream LLaMA-Factory
# does not dispatch liger for gemma4/gemma4_text as of the pinned base
# commit, and enable_liger_kernel: true silently falls through to a
# "does not support liger kernel" no-op warning without it.
#
# default_cutoff_len=28672: real max tokenized length of validated training
# data under Gemma-4's tokenizer was measured at 27,979 (p99=20,549).
# Gemma-4 has materially higher per-token activation memory than
# similarly-sized dense models, so a cutoff_len that's safe for Qwen3 is
# not automatically safe here -- this default, plus flash_attn+liger
# above, is what made a real run fit in 80GB.
_GEMMA4 = FamilyConfig(
    template="gemma4",
    think_open="<|channel>thought\n",
    think_close="<channel|>",
    freeze_vision=True,
    flash_attn="fa2",
    enable_liger_kernel=True,
    default_cutoff_len=28672,
)

_FAMILIES = {
    "qwen3": _QWEN3,
    "gemma4": _GEMMA4,
}


def get(model_family: str) -> FamilyConfig:
    try:
        return _FAMILIES[model_family]
    except KeyError:
        raise ValueError(
            f"Unsupported model_family {model_family!r}. HETU_Online only "
            f"supports {MODEL_FAMILIES} -- see model_families.py module "
            f"docstring for why other families aren't just a config away."
        ) from None


def think_tags(model_family: str) -> Tuple[str, str]:
    cfg = get(model_family)
    return cfg.think_open, cfg.think_close


def template(model_family: str) -> str:
    return get(model_family).template
