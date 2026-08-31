"""
Runs one (checkpoint, benchmark) eval job: loads the base model + LoRA
adapter in vLLM, generates a response for every question in the benchmark,
repeats the whole benchmark N_REPEATS times (default 3, per-run generation
is stochastic via sampling -- see NOTE below), grades every response,
splits each response into its <think>...</think> reasoning trace and final
answer text, and writes:
  - a full per-query JSONL trace (one line per (repeat, question)) with the
    prompt, complete raw response, extracted thinking, extracted final
    answer, extracted graded answer, correctness, and per-query token
    counts/timing
  - a summary JSON with per-repeat and aggregate (mean/std across repeats)
    accuracy, throughput (input/output tokens per second), reasoning-token
    length stats, wall-clock time, and queries/sec

The core logic lives in run_eval(cfg: EvalConfig) so both this module's CLI
entrypoint (`hetu-online eval`) and any interactive/notebook use call the
exact same code path.

NOTE on determinism: greedy (temperature=0) decoding would make all 3
repeats byte-identical, which defeats the purpose of repeating -- the
"averaged over 3 runs" figure is only meaningful under sampling. Sampling
params come from each model's own HF card (see
eval_model_families.SAMPLING_PARAMS_BY_FAMILY) so repeats actually differ.

NOTE on truncated generations (hit max_tokens before reaching </think> +
an answer): rather than forcibly injecting "</think>" and asking for just
the answer (which would score the model on an answer it never actually
reached on its own), or fully discarding+regenerating from scratch
(doesn't guarantee the retry fits either, and throws away reasoning
progress already paid for), a truncated response is CONTINUED: the exact
prompt+partial-generation-so-far is fed back in as a new prompt with a
fresh, larger token budget, letting the model finish its own reasoning and
then naturally emit </think> + an answer. This can repeat up to
max_continuation_rounds times if it keeps hitting the budget. Continued
rows are flagged `was_truncated: true` in the trace and contribute to a
SEPARATE accuracy figure (`accuracy_mean_excl_extended` in the summary,
alongside the normal `accuracy_mean` which counts them using whatever
answer the extended generation eventually reached) -- both are written to
the summary and both flow into the final comparison table, since a
truncated-then-forced-to-continue answer is a meaningfully different
condition from "answered within the original budget" and should not be
silently blended into it.

Usage (CLI, via the package entrypoint):
  hetu-online eval --checkpoint_dir <path> --model_family <qwen3|dsr1_llama8b|dsr1_qwen7b|...> \\
      --dataset_name <name> --mode <cotgen|cotcond|base> \\
      --benchmark <aime24|aime25|math500|gpqa_diamond> \\
      --output_dir <dir> [--num_repeats 3] [--max_new_tokens 32768] [--max_samples N]

GPU selection is via CUDA_VISIBLE_DEVICES (or --gpu_id, set before any CUDA
init happens).
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .benchmarks import BENCHMARKS, format_question, grade, load_benchmark
from .eval_model_families import (
    MODEL_FAMILIES,
    NATIVE_MAX_CONTEXT,
    base_model_path,
    build_chat_messages,
    continuation_max_new_tokens_for,
    forced_assistant_prefix,
    max_continuation_rounds_for,
    needs_enable_thinking,
    sampling_params_for,
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class EvalConfig:
    model_family: str  # "qwen3" | "dsr1_llama8b" | "dsr1_qwen7b" | "qwen3_0p6b" | "qwen3_4b"
    dataset_name: str
    mode: str  # "cotgen" | "cotcond" | "base"
    benchmark: str
    output_dir: str
    checkpoint_dir: Optional[str] = None  # None + mode="base" -> no LoRA adapter
    num_repeats: int = 3
    max_new_tokens: int = 32768
    continuation_max_new_tokens: Optional[int] = None  # None -> per-family default (see eval_model_families.py)
    max_continuation_rounds: Optional[int] = None  # None -> per-family default (see eval_model_families.py)
    temperature: Optional[float] = None  # None -> use model card default
    top_p: Optional[float] = None  # None -> use model card default
    seed_base: int = 12345
    max_samples: Optional[int] = None  # None -> full benchmark; else first N questions
    gpu_memory_utilization: float = 0.90
    log_fn: callable = field(default=print)  # swap for a notebook-friendly logger
    show_progress: bool = True  # tqdm bars over repeats/continuation rounds
    run_id: Optional[str] = None  # None -> default run_name (overwrites any prior run of the
    # same dataset_name/mode/benchmark). Set to a distinct tag (e.g. "manual1", "nb2") to give
    # a separate invocation its own {run_name}_{run_id}_summary.json/traces.jsonl instead of
    # silently clobbering a previous run's output -- see tables.py, which globs every
    # eval_results/*_summary.json file regardless of this suffix, so multiple tagged runs of
    # the same config all show up as separate rows in the final table.

    def __post_init__(self):
        if self.mode == "base" and self.checkpoint_dir is not None:
            raise ValueError("checkpoint_dir must be None when mode='base' (base-model eval has no adapter)")
        if self.mode != "base" and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required unless mode='base'")
        # Per-family defaults, NOT a single shared default: Qwen3-8B's
        # native context (40960) cannot fit the same 2-round/32768-per-round
        # continuation budget that DeepSeek-distill (131072 context) can --
        # see eval_model_families.py's MAX_CONTINUATION_ROUNDS_BY_FAMILY /
        # CONTINUATION_MAX_NEW_TOKENS_OVERRIDE for the exact numbers and why.
        if self.max_continuation_rounds is None:
            self.max_continuation_rounds = max_continuation_rounds_for(self.model_family)
        if self.continuation_max_new_tokens is None:
            self.continuation_max_new_tokens = continuation_max_new_tokens_for(
                self.model_family, self.max_new_tokens
            )

    def required_max_model_len(self, prompt_headroom: int = 4096) -> int:
        """vLLM's max_model_len must cover the WORST-CASE cumulative sequence
        length across every continuation round, not just one round's budget:
        each continuation re-feeds (original prompt + everything generated so
        far) as the new prompt, so by the final round the "prompt" alone can
        already equal prompt_headroom + max_new_tokens +
        (max_continuation_rounds - 1) * continuation_max_new_tokens tokens,
        and vLLM validates prompt_len + requested_max_tokens against
        max_model_len BEFORE generating a single token of that round. Missing
        this caused a real failure: a GPQA-Diamond eval died mid-run with
        "decoder prompt (length 36864) plus requested output tokens is longer
        than the maximum model length of 36864" the first time a response
        needed a second continuation round, because max_model_len was only
        sized for max_new_tokens + prompt_headroom (one round), not the full
        multi-round worst case.

        Clamped to the model's own NATIVE_MAX_CONTEXT (e.g. Qwen3-8B's
        40960): requesting a max_model_len vLLM's model config can't
        actually support fails at engine-init time, not gracefully, so this
        must never exceed what the architecture allows -- the per-family
        continuation-round defaults in eval_model_families.py are chosen
        specifically so the unclamped total already fits, and this clamp is
        a defensive floor/ceiling, not the primary mechanism keeping
        requests in bounds."""
        native_ceiling = NATIVE_MAX_CONTEXT[self.model_family]
        required = prompt_headroom + self.max_new_tokens + self.max_continuation_rounds * self.continuation_max_new_tokens
        return min(required, native_ceiling)


def split_think_answer(prefix: str, generated_text: str) -> tuple[str, str]:
    """`prefix` is the reasoning-opener the PROMPT already ended with (see
    forced_assistant_prefix's docstring): "<think>\\n" for DeepSeek-distill
    (baked into its raw chat template), "" for Qwen3 (nothing pre-opened,
    the model emits its own <think> from scratch). Reconstructing
    prefix + generated_text gives the regex a complete <think>...</think>
    span to find in both cases, without ever re-appending a tag that was
    already part of the prompt vLLM was given."""
    full_text = prefix + generated_text
    m = THINK_RE.search(full_text)
    if m is None:
        return "", full_text.strip()
    thinking = m.group(1).strip()
    answer = full_text[m.end():].strip()
    return thinking, answer


def generate_with_continuation(
    llm,
    lora_request,
    prompts: list[str],
    sampling_kwargs: dict,
    max_new_tokens: int,
    continuation_max_new_tokens: int,
    max_continuation_rounds: int,
    tokenizer,
    max_model_len: int,
    log_fn=print,
    show_progress: bool = True,
):
    """Runs one generation pass over all prompts, then for any row whose
    finish_reason is "length" (hit the token budget, not a natural stop),
    re-generates just that row's continuation by feeding
    prompt + text-so-far back in as the new prompt, up to
    max_continuation_rounds times. Returns a list of dicts, one per input
    prompt, in the same order:
        {"text": full generated text (all rounds concatenated),
         "initial_round_text": text from round 0 ONLY (what a
                                no-continuation eval would have seen/graded),
         "prompt_tokens": int (from the ORIGINAL prompt only),
         "output_tokens": int (summed across all rounds for this row),
         "initial_round_output_tokens": int (round 0 only),
         "continuation_output_tokens": int (output_tokens minus
                                             initial_round_output_tokens --
                                             i.e. how many extra tokens the
                                             continuation rounds needed),
         "was_truncated": bool (True if it ever hit "length" and needed
                                 at least one continuation round),
         "still_truncated": bool (True if it STILL hadn't naturally
                                   stopped after max_continuation_rounds --
                                   a hard failure to produce an answer even
                                   with extended budget)}
    """
    from vllm import SamplingParams

    n = len(prompts)
    texts = [""] * n
    initial_round_texts = [""] * n
    output_token_counts = [0] * n
    initial_round_output_tokens = [0] * n
    prompt_token_counts = [0] * n
    was_truncated = [False] * n
    still_truncated = [False] * n
    still_pending = list(range(n))

    round_prompts = list(prompts)
    round_max_tokens = max_new_tokens

    for round_idx in range(1 + max_continuation_rounds):
        if not still_pending:
            break

        round_label = "initial" if round_idx == 0 else f"continuation {round_idx}"
        log_fn(f"    [generate] {round_label} round: {len(still_pending)} prompt(s), max_tokens={round_max_tokens}")

        batch_prompts = [round_prompts[i] for i in still_pending]
        # Per-row clamp: by a later continuation round, round_prompts[idx]
        # already contains the original prompt PLUS everything generated in
        # every prior round, so its token length can vary a lot per row
        # (some rows may have stopped naturally in round 0, others may have
        # used their full budget every round so far). vLLM validates
        # prompt_len + requested_max_tokens against max_model_len BEFORE
        # generating anything and raises if it doesn't fit -- so max_tokens
        # must be clamped per-row here rather than using one shared value
        # for the whole batch (a single shared max_tokens is exactly what
        # caused a real "decoder prompt (length 36864) plus requested output
        # tokens is longer than max_model_len" failure on a real eval run).
        sampling_params_list = []
        for i in still_pending:
            prompt_len = len(tokenizer.encode(round_prompts[i], add_special_tokens=False))
            budget_left = max(1, max_model_len - prompt_len)
            this_max_tokens = min(round_max_tokens, budget_left)
            sampling_params_list.append(SamplingParams(max_tokens=this_max_tokens, **sampling_kwargs))
        outputs = llm.generate(
            batch_prompts, sampling_params_list, lora_request=lora_request, use_tqdm=show_progress
        )

        next_pending = []
        num_hit_limit_this_round = 0
        for idx, output in zip(still_pending, outputs):
            gen = output.outputs[0]
            if round_idx == 0:
                prompt_token_counts[idx] = len(output.prompt_token_ids)
                initial_round_texts[idx] = gen.text
                initial_round_output_tokens[idx] = len(gen.token_ids)
            texts[idx] += gen.text
            output_token_counts[idx] += len(gen.token_ids)

            hit_length_limit = gen.finish_reason == "length"
            still_truncated[idx] = hit_length_limit
            if hit_length_limit:
                num_hit_limit_this_round += 1
                was_truncated[idx] = True
                if round_idx < max_continuation_rounds:
                    round_prompts[idx] = round_prompts[idx] + gen.text
                    next_pending.append(idx)
                # else: exhausted continuation-round budget; still_truncated[idx] stays True

        log_fn(f"    [generate] {round_label} round done: {num_hit_limit_this_round} hit the token limit")
        still_pending = next_pending
        round_max_tokens = continuation_max_new_tokens

    return [
        {
            "text": texts[i],
            "initial_round_text": initial_round_texts[i],
            "prompt_tokens": prompt_token_counts[i],
            "output_tokens": output_token_counts[i],
            "initial_round_output_tokens": initial_round_output_tokens[i],
            "continuation_output_tokens": output_token_counts[i] - initial_round_output_tokens[i],
            "was_truncated": was_truncated[i],
            "still_truncated": still_truncated[i],
        }
        for i in range(n)
    ]


def build_prompt(tokenizer, model_family: str, question: str) -> str:
    """Both families' raw chat templates already auto-append the reasoning
    opener at add_generation_prompt time (DeepSeek-distill: literal
    "<think>\\n"; Qwen3 with enable_thinking=True: nothing pre-opened, model
    emits its own <think>). forced_assistant_prefix() documents what's
    already in `text` for split_think_answer() to use later -- it must NOT
    be concatenated again here, or DeepSeek prompts would end up with the
    <think> tag duplicated."""
    messages = build_chat_messages(model_family, question)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if needs_enable_thinking(model_family):
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_llm_and_tokenizer(cfg: EvalConfig):
    """Loads the vLLM engine + base-model tokenizer for cfg. Split out from
    run_eval() so a caller can load once and reuse the engine across
    multiple evals of the SAME model_family (different checkpoint_dir/mode)
    without repaying vLLM's ~1-2 min startup cost each time -- though
    switching model_family still requires a fresh engine (different base
    model weights)."""
    from transformers import AutoTokenizer
    from vllm import LLM

    base_path = base_model_path(cfg.model_family)
    # Load the tokenizer from the BASE model, not the checkpoint dir: LoRA
    # never changes the vocabulary/chat template, and the checkpoint's saved
    # tokenizer_config.json may be written by a newer/older transformers
    # version than this eval environment's, which can fail to parse it
    # (confirmed: training env's transformers 5.6.0 renames a key that eval
    # env's 4.57.6 expects under its older name).
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    llm = LLM(
        model=base_path,
        enable_lora=True,  # kept enabled even for base-model evals so the same
        # engine instance can be reused for a later LoRA eval of this family
        # without reloading; passing lora_request=None per-call still runs
        # the un-adapted base model.
        max_lora_rank=32,
        max_model_len=cfg.required_max_model_len(),
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    return llm, tokenizer


_AUTO_RUN_ID_RE_TEMPLATE = r"^{base}_run(\d+)_summary\.json$"


def _next_auto_run_id(out_dir: Path, base_run_name: str) -> str:
    """Picks the next unused "runN" tag for this exact
    (dataset_name, mode, benchmark) so that NOT passing --run_id/run_id never
    overwrites a prior run's output by default -- every invocation gets its
    own numbered slot (run1, run2, run3, ...) unless the caller explicitly
    passes a run_id to use instead. Scans out_dir for existing
    "{base_run_name}_runN_summary.json" files and returns "run{max(N)+1}"
    (or "run1" if none exist yet). Also treats a pre-existing *unsuffixed*
    "{base_run_name}_summary.json" (from before this auto-numbering existed,
    or from an explicit run_id that happens not to look like "runN") as
    already-occupied ground -- it's left alone, never renamed or counted
    into the numbering, but its presence doesn't block run1..N from being
    assigned since it doesn't match the runN pattern being scanned for."""
    pattern = re.compile(_AUTO_RUN_ID_RE_TEMPLATE.format(base=re.escape(base_run_name)))
    max_n = 0
    if out_dir.exists():
        for p in out_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"run{max_n + 1}"


def _build_summary(
    cfg: EvalConfig,
    run_id: str,
    rows: list,
    per_repeat_stats: list,
    sampling_kwargs_base: dict,
    trace_path: Path,
    is_complete: bool,
    total_elapsed: float,
) -> dict:
    """Aggregates whatever repeats are in per_repeat_stats so far into one
    summary dict -- called after EVERY repeat (not just the last one), so a
    table-building pass reading this file mid-run sees real numbers averaged
    over however many repeats actually completed, marked via
    num_repeats_completed/is_complete rather than silently understating
    num_repeats or leaving the file absent until everything finishes."""
    accuracies = [r["accuracy"] for r in per_repeat_stats]
    accuracies_excl_extended = [r["accuracy_excl_extended"] for r in per_repeat_stats]
    output_tps = [r["output_tokens_per_sec"] for r in per_repeat_stats]
    input_tps = [r["input_tokens_per_sec"] for r in per_repeat_stats]
    qps = [r["queries_per_sec"] for r in per_repeat_stats]
    reasoning_lens = [r["mean_reasoning_tokens"] for r in per_repeat_stats]

    return {
        "dataset_name": cfg.dataset_name,
        "mode": cfg.mode,
        "model_family": cfg.model_family,
        "benchmark": cfg.benchmark,
        "run_id": run_id,
        "checkpoint_dir": cfg.checkpoint_dir,
        "num_repeats": cfg.num_repeats,
        "num_repeats_completed": len(per_repeat_stats),
        "is_complete": is_complete,
        "num_questions": len(rows),
        "max_samples": cfg.max_samples,
        "sampling_params": sampling_kwargs_base,
        "max_new_tokens": cfg.max_new_tokens,
        "continuation_max_new_tokens": cfg.continuation_max_new_tokens,
        "max_continuation_rounds": cfg.max_continuation_rounds,
        "total_wall_clock_seconds": total_elapsed,
        "per_repeat": per_repeat_stats,
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "accuracy_mean_excl_extended": statistics.mean(accuracies_excl_extended),
        "accuracy_std_excl_extended": (
            statistics.stdev(accuracies_excl_extended) if len(accuracies_excl_extended) > 1 else 0.0
        ),
        "num_extended_total": sum(r["num_extended"] for r in per_repeat_stats),
        "num_extended_correct_total": sum(r["num_extended_correct"] for r in per_repeat_stats),
        "num_extended_incorrect_total": sum(r["num_extended_incorrect"] for r in per_repeat_stats),
        "extended_accuracy_mean": (
            statistics.mean([r["extended_accuracy"] for r in per_repeat_stats if r["extended_accuracy"] is not None])
            if any(r["extended_accuracy"] is not None for r in per_repeat_stats)
            else None
        ),
        "num_still_truncated_total": sum(r["num_still_truncated"] for r in per_repeat_stats),
        "mean_extended_reasoning_tokens_added_mean": statistics.mean(
            [r["mean_extended_reasoning_tokens_added"] for r in per_repeat_stats]
        ),
        "output_tokens_per_sec_mean": statistics.mean(output_tps),
        "input_tokens_per_sec_mean": statistics.mean(input_tps),
        "queries_per_sec_mean": statistics.mean(qps),
        "mean_reasoning_tokens_mean": statistics.mean(reasoning_lens),
        "trace_file": str(trace_path),
    }


def run_eval(cfg: EvalConfig, llm=None, tokenizer=None) -> dict:
    """Runs the full (cfg.num_repeats)-repeat eval described by cfg and
    returns the summary dict (also written to
    {cfg.output_dir}/{run_name}_summary.json, with the full trace written to
    {cfg.output_dir}/{run_name}_traces.jsonl). Pass an already-loaded
    (llm, tokenizer) pair (from load_llm_and_tokenizer) to reuse an engine
    across multiple calls for the same model_family; otherwise one is
    loaded and torn down within this call."""
    from vllm.lora.request import LoRARequest

    log = cfg.log_fn
    owns_engine = llm is None
    if owns_engine:
        log(f"[eval] loading vLLM engine for model_family={cfg.model_family} ...")
        llm, tokenizer = load_llm_and_tokenizer(cfg)
        log("[eval] engine loaded.")

    sampling_kwargs_base = sampling_params_for(cfg.model_family)
    if cfg.temperature is not None:
        sampling_kwargs_base["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        sampling_kwargs_base["top_p"] = cfg.top_p
    log(f"[eval] sampling params: {sampling_kwargs_base}")

    lora_request = LoRARequest("adapter", 1, cfg.checkpoint_dir) if cfg.checkpoint_dir is not None else None

    rows = load_benchmark(cfg.benchmark)
    if cfg.max_samples is not None:
        rows = rows[: cfg.max_samples]
    log(f"[eval] benchmark={cfg.benchmark} -- {len(rows)} question(s) (max_samples={cfg.max_samples})")
    prompts = [build_prompt(tokenizer, cfg.model_family, format_question(cfg.benchmark, r)) for r in rows]

    base_run_name = f"{cfg.dataset_name}_{cfg.mode}_{cfg.benchmark}"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = cfg.run_id
    if run_id is None:
        run_id = _next_auto_run_id(out_dir, base_run_name)
        log(f"[eval] no --run_id given -- auto-assigned {run_id!r} for {base_run_name} "
            f"(existing runs of this exact dataset_name/mode/benchmark are never overwritten by default)")

    run_name = f"{base_run_name}_{run_id}"
    trace_path = out_dir / f"{run_name}_traces.jsonl"
    summary_path = out_dir / f"{run_name}_summary.json"

    per_repeat_stats = []
    trace_f = open(trace_path, "w")

    prefix = forced_assistant_prefix(cfg.model_family)

    repeat_iter = range(cfg.num_repeats)
    if cfg.show_progress:
        from tqdm.auto import tqdm

        repeat_iter = tqdm(repeat_iter, desc=f"{run_name} repeats", unit="repeat")

    overall_start = time.time()
    for repeat_idx in repeat_iter:
        log(f"[eval] {run_name} -- starting repeat {repeat_idx + 1}/{cfg.num_repeats}")
        sampling_kwargs_this_repeat = dict(sampling_kwargs_base, seed=cfg.seed_base + repeat_idx)

        repeat_start = time.time()
        gen_results = generate_with_continuation(
            llm,
            lora_request,
            prompts,
            sampling_kwargs_this_repeat,
            cfg.max_new_tokens,
            cfg.continuation_max_new_tokens,
            cfg.max_continuation_rounds,
            tokenizer,
            cfg.required_max_model_len(),
            log_fn=log,
            show_progress=cfg.show_progress,
        )
        repeat_elapsed = time.time() - repeat_start

        num_correct = 0
        num_correct_excl_extended = 0
        num_extended = 0
        num_extended_correct = 0
        num_extended_incorrect = 0
        num_still_truncated = 0
        total_prompt_tokens = 0
        total_output_tokens = 0
        reasoning_token_counts = []
        extended_reasoning_added_counts = []

        for row, prompt_text, gen_result in zip(rows, prompts, gen_results):
            generated_text = gen_result["text"]
            thinking, answer_text = split_think_answer(prefix, generated_text)
            is_correct, extracted = grade(cfg.benchmark, answer_text, row["reference"])

            prompt_token_count = gen_result["prompt_tokens"]
            output_token_count = gen_result["output_tokens"]
            reasoning_token_count = len(tokenizer.encode(thinking, add_special_tokens=False)) if thinking else 0

            total_prompt_tokens += prompt_token_count
            total_output_tokens += output_token_count
            reasoning_token_counts.append(reasoning_token_count)
            if is_correct:
                num_correct += 1
            # "excl_extended" is the honest within-original-budget figure:
            # any row that needed continuation counts as incorrect here,
            # regardless of what the extended generation eventually answered.
            was_extended = gen_result["was_truncated"]
            if was_extended:
                num_extended += 1
                if gen_result["still_truncated"]:
                    num_still_truncated += 1
                if is_correct:
                    num_extended_correct += 1
                else:
                    num_extended_incorrect += 1
                # Reasoning-token length attributable to the continuation
                # rounds alone: reasoning found in the FULL text minus
                # reasoning already present in the initial-round-only text
                # (both computed the same way, via split_think_answer, so a
                # negative value only happens if </think> already appeared
                # in round 0 -- shouldn't occur since was_truncated implies
                # round 0 hit "length" before any stop token, but clamped to
                # 0 defensively).
                initial_thinking, _ = split_think_answer(prefix, gen_result["initial_round_text"])
                initial_reasoning_tokens = (
                    len(tokenizer.encode(initial_thinking, add_special_tokens=False)) if initial_thinking else 0
                )
                extended_reasoning_added_counts.append(max(0, reasoning_token_count - initial_reasoning_tokens))
            elif is_correct:
                num_correct_excl_extended += 1

            trace_f.write(
                json.dumps(
                    {
                        "repeat": repeat_idx,
                        "benchmark": cfg.benchmark,
                        "question_id": row["id"],
                        "question": row["question"],
                        "choices": row.get("choices"),
                        "reference": row["reference"],
                        "prompt": prompt_text,
                        "raw_response": generated_text,
                        "thinking": thinking,
                        "final_answer_text": answer_text,
                        "extracted_answer": extracted,
                        "is_correct": is_correct,
                        "was_truncated": gen_result["was_truncated"],
                        "still_truncated": gen_result["still_truncated"],
                        "prompt_tokens": prompt_token_count,
                        "output_tokens": output_token_count,
                        "continuation_output_tokens": gen_result["continuation_output_tokens"],
                        "reasoning_tokens": reasoning_token_count,
                    }
                )
                + "\n"
            )
        trace_f.flush()

        accuracy = num_correct / len(rows)
        accuracy_excl_extended = num_correct_excl_extended / len(rows)
        input_tok_per_sec = total_prompt_tokens / repeat_elapsed
        output_tok_per_sec = total_output_tokens / repeat_elapsed
        queries_per_sec = len(rows) / repeat_elapsed
        extended_accuracy = (num_extended_correct / num_extended) if num_extended else None

        per_repeat_stats.append(
            {
                "repeat": repeat_idx,
                "accuracy": accuracy,
                "accuracy_excl_extended": accuracy_excl_extended,
                "num_correct": num_correct,
                "num_correct_excl_extended": num_correct_excl_extended,
                "num_extended": num_extended,
                "num_extended_correct": num_extended_correct,
                "num_extended_incorrect": num_extended_incorrect,
                "extended_accuracy": extended_accuracy,
                "num_still_truncated": num_still_truncated,
                "num_questions": len(rows),
                "elapsed_seconds": repeat_elapsed,
                "total_prompt_tokens": total_prompt_tokens,
                "total_output_tokens": total_output_tokens,
                "input_tokens_per_sec": input_tok_per_sec,
                "output_tokens_per_sec": output_tok_per_sec,
                "queries_per_sec": queries_per_sec,
                "mean_reasoning_tokens": statistics.mean(reasoning_token_counts) if reasoning_token_counts else 0,
                "median_reasoning_tokens": statistics.median(reasoning_token_counts) if reasoning_token_counts else 0,
                "mean_extended_reasoning_tokens_added": (
                    statistics.mean(extended_reasoning_added_counts) if extended_reasoning_added_counts else 0
                ),
                "median_extended_reasoning_tokens_added": (
                    statistics.median(extended_reasoning_added_counts) if extended_reasoning_added_counts else 0
                ),
            }
        )
        extended_acc_str = f"{extended_accuracy:.4f}" if extended_accuracy is not None else "n/a"
        log(
            f"[eval] {run_name} repeat {repeat_idx}: acc={accuracy:.4f} "
            f"acc_excl_extended={accuracy_excl_extended:.4f} "
            f"({num_correct}/{len(rows)}, {num_extended} extended [{num_extended_correct} correct / "
            f"{num_extended_incorrect} incorrect, acc={extended_acc_str}], "
            f"{num_still_truncated} still_truncated) elapsed={repeat_elapsed:.1f}s "
            f"out_tok/s={output_tok_per_sec:.1f}"
        )

        # Write/overwrite the summary after EVERY repeat, not just once at the
        # end -- so a table-building pass run at any moment sees however many
        # repeats have actually landed for this run (1, 2, or all num_repeats),
        # and a job that dies partway through (OOM, node preemption, wall-time
        # limit) still leaves every repeat it DID finish safely on disk instead
        # of losing the whole run. is_complete/num_repeats_completed let a
        # table builder distinguish a still-accumulating run from a finished
        # one without guessing from wall-clock time.
        is_complete = (repeat_idx + 1) == cfg.num_repeats
        summary = _build_summary(
            cfg,
            run_id,
            rows,
            per_repeat_stats,
            sampling_kwargs_base,
            trace_path,
            is_complete,
            time.time() - overall_start,
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"[eval] wrote summary ({repeat_idx + 1}/{cfg.num_repeats} repeats done) to {summary_path}")

    trace_f.close()

    if owns_engine:
        # Free GPU memory for a subsequent eval in the same process (back-to-back
        # runs with different model_families in one long-lived process).
        import gc

        import torch

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return summary


def add_eval_arguments(ap: argparse.ArgumentParser) -> None:
    """Adds all `hetu-online eval` flags to an existing subparser -- split
    out so cli.py can register this subcommand without duplicating the
    flag list."""
    ap.add_argument(
        "--checkpoint_dir",
        default=None,
        help="LoRA adapter checkpoint dir. Omit to evaluate the un-fine-tuned base model directly.",
    )
    ap.add_argument("--model_family", required=True, choices=list(MODEL_FAMILIES))
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--mode", required=True, choices=["cotgen", "cotcond", "base"])
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_repeats", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=32768)
    ap.add_argument(
        "--continuation_max_new_tokens",
        type=int,
        default=None,
        help="Token budget for each continuation round after an initial truncation. Defaults to the "
        "per-model_family value in eval_model_families.py (see CONTINUATION_MAX_NEW_TOKENS_OVERRIDE) if not set.",
    )
    ap.add_argument(
        "--max_continuation_rounds",
        type=int,
        default=None,
        help="How many extra continuation rounds to allow a truncated response before giving up. Defaults to the "
        "per-model_family value in eval_model_families.py (see MAX_CONTINUATION_ROUNDS_BY_FAMILY) if not set -- do "
        "NOT hardcode a shared default here: Qwen3-8B's native context (40960) cannot fit the same "
        "2-round/32768-per-round budget DeepSeek-distill's 131072 context can (this exact mistake previously "
        "caused a real eval crash).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Overrides the model card's recommended temperature for --model_family if set.",
    )
    ap.add_argument("--top_p", type=float, default=None, help="Overrides the model card's recommended top_p if set.")
    ap.add_argument("--seed_base", type=int, default=12345)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N questions of the benchmark instead of the full set.",
    )
    ap.add_argument("--gpu_id", type=str, default=None, help="Sets CUDA_VISIBLE_DEVICES before any CUDA init.")
    ap.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Distinct tag for this invocation (e.g. 'manual1', 'nb2'). When set, output files become "
        "{run_name}_{run_id}_summary.json/traces.jsonl instead of {run_name}_summary.json/traces.jsonl, so "
        "repeated separate invocations of the SAME dataset_name/mode/benchmark don't silently overwrite each "
        "other's results. tables.py globs every *_summary.json file regardless of this suffix, so each tagged "
        "run shows up as its own row in the final table.",
    )


def run_eval_from_args(args: argparse.Namespace) -> int:
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    cfg = EvalConfig(
        model_family=args.model_family,
        dataset_name=args.dataset_name,
        mode=args.mode,
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_repeats=args.num_repeats,
        max_new_tokens=args.max_new_tokens,
        continuation_max_new_tokens=args.continuation_max_new_tokens,
        max_continuation_rounds=args.max_continuation_rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        seed_base=args.seed_base,
        max_samples=args.max_samples,
        run_id=args.run_id,
    )
    run_eval(cfg)
    return 0


def main(argv=None) -> int:
    """Standalone entrypoint (python -m hetu_online.eval.run_eval ...),
    kept for parity with the original script-based invocation; `hetu-online
    eval` (cli.py) is the normal way to run this."""
    ap = argparse.ArgumentParser()
    add_eval_arguments(ap)
    args = ap.parse_args(argv)
    return run_eval_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
