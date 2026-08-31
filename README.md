# HETU_Online

A small, pip-installable tool for training a model two ways on the same
reasoning data — **CotGen** and **CotCond** — on top of
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Give it a base
model, a `--model_family`, and a JSON file of `{prompt, thinking, answer}`
examples, and it produces ready-to-run LoRA SFT configs for both modes, plus
the training-time loss masking CotCond needs (which stock LLaMA-Factory
can't do on its own).

**CLI-enabled model family: `qwen3` (any size) only, for now.** `gemma4`
(E2B/E4B/12B) has a fully verified `model_families.py` entry, a vendored
LLaMA-Factory patch, and its own tests, but has not yet completed a
validated end-to-end training run on real GPU hardware — see "Gemma-4:
code shipped, not yet CLI-enabled" below. `--model_family` is a closed
choice, not a free-text template name, because a wrong template or
think-tag pair for an unvalidated model fails **silently** (CotCond's
masking fails open to full supervision with no error). Adding a new
family means adding a verified entry to `model_families.py`, not just
passing a different string — and even a verified entry stays CLI-blocked
until it has a real successful training run behind it (see
`cli.py`'s `CLI_ENABLED_MODEL_FAMILIES`).

This implements what's referred to as **"Family B"** below — the model
generates its **own** `<think>...</think>` reasoning at both train and
inference time in both modes. The two modes differ only in whether the
reasoning *content* gets gradient:

| Mode | Response the model is trained to produce | What gets gradient |
|---|---|---|
| **CotGen** | `<think>{reasoning}</think>\n\n{answer}` | Everything. The model imitates the teacher's reasoning **and** answer directly. |
| **CotCond** | `<think>{reasoning}</think>\n\n{answer}` (same text) | Only the `<think>`/`</think>` tag tokens and the answer. The reasoning **content** between the tags is masked to zero loss. |

### Why CotCond isn't "teach it to think empty thoughts"

Masking the content span doesn't reward brevity or silence — there's no
length penalty anywhere in the objective. Every training example still has
realistic-length reasoning between the tags, so the model still learns
"after `<think>`, plausible reasoning-shaped text is expected, not an
immediate close." What's absent is any pull toward the *specific wording*
of this corpus's reasoning. Two things are enforced structurally (always
open a think block, always close it before answering); the content in
between is free-running, sampled from the base model's own next-token
distribution, never directly supervised.

The research question CotCond answers: **"if the model has to condition
its final answer on reasoning it generated itself (not a teacher's), does
it still get the answer right?"** — as opposed to CotGen, which only tells
you the model can imitate a teacher's reasoning, never whether its answer
actually depends on that reasoning being any good.

---

## Install

This repo covers two separate, deliberately non-overlapping environments —
training (LLaMA-Factory + LoRA SFT) and eval (vLLM generation + grading).
They were validated as two separate conda envs with different Python/CUDA
requirements; installing both into one env is not tested and may hit
dependency conflicts (in particular, torch/transformers versions that
training and vLLM want tend to diverge).

**Trainer environment:**

```bash
git clone <this-repo-url> HETU_Online
cd HETU_Online
pip install -e .

# LLaMA-Factory is the training backend and is a separate install --
# do this once, in the same environment:
pip install -e /path/to/LLaMA-Factory
```

**Eval environment** (separate env from the trainer one):

```bash
cd HETU_Online
pip install -e ".[eval]"   # datasets, vllm, transformers, tqdm, pandas
```

Check your GPU's CUDA/driver version before installing `vllm`/`torch` in
either environment — the right build isn't pinned here since it depends on
the machine, and a mismatch can silently install a broken or CPU-only
build (see "Gemma-4 prerequisites" below for the training side's own
version-sensitivity note).

Either install gives you the `hetu-online` command from anywhere — no need
to `cd` into the repo or remember script order.

---

## Quickstart

```bash
# 1. Build both datasets from one raw file (prompt/thinking/answer JSON)
hetu-online build-data \
  --input /path/to/raw_train.json \
  --out_dir /path/to/LLaMA-Factory/data/my_task \
  --dataset_name my_task \
  --llamafactory_data_dir /path/to/LLaMA-Factory/data \
  --model_family qwen3 \
  --system_prompt "You are a careful, helpful assistant."

# 2. Generate a config for each mode -- model_family fills in template,
#    freeze_vision_tower/flash_attn/enable_liger_kernel (gemma4 only), and
#    the validated default cutoff_len for you.
hetu-online make-config --dataset_name my_task --mode cotgen  \
  --model_path /path/to/base_model --output_dir /path/to/checkpoints/my_task/cotgen \
  --model_family qwen3

hetu-online make-config --dataset_name my_task --mode cotcond \
  --model_path /path/to/base_model --output_dir /path/to/checkpoints/my_task/cotcond \
  --model_family qwen3

# 3. Train -- --model_family must come BEFORE the positional args
hetu-online train --model_family qwen3 my_task cotgen  configs/my_task_cotgen.yaml
hetu-online train --model_family qwen3 my_task cotcond configs/my_task_cotcond.yaml
```

Gemma-4 is not yet a CLI-accepted `--model_family` (see above) — the code
and config are in this repo for when GPU validation completes, but
`--model_family gemma4` is refused by argparse today, not silently
accepted.

`hetu-online train` handles the one thing that's easy to forget: CotCond
needs an environment variable set at launch (`HETU_THINK_CONTENT_MASK=1`),
not a YAML key, because LLaMA-Factory's stock dataset processor can only
mask at whole-turn granularity and has no concept of masking a sub-span
inside one turn. Forgetting that env var doesn't error — it silently
trains full-loss instead, which is the whole reason this tool wraps the
launch command instead of leaving it to you to remember.

Multi-GPU works the same as any other `llamafactory-cli` run — set
`CUDA_VISIBLE_DEVICES` (and your accelerate config, if any) in your shell
before calling `hetu-online train`.

---

## Input data format

One JSON array (or `.jsonl`) of examples, each with exactly these three fields:

```json
[
  {
    "prompt": "What is the capital of France?",
    "thinking": "France is a country in Europe. Its capital city, and largest city, is Paris.",
    "answer": "Paris"
  }
]
```

- `prompt` — the raw user-turn instruction. No hint text, no reasoning
  pasted in — that's what makes this Family B, not Family A (see
  "Related approaches" below).
- `thinking` — the teacher's reasoning trace. Used identically for both
  CotGen and CotCond; the two modes never see different *text*, only
  different *masking*.
- `answer` — the final answer text.

If you don't have a val split, `build-data` holds out a random slice of
`--input` for you (`--val_fraction`, default 2%). Pass `--val_input`
explicitly if you already have one.

---

## Commands

### `hetu-online build-data`
Raw `{prompt, thinking, answer}` → ShareGPT JSON for both modes, registers
them in LLaMA-Factory's `dataset_info.json`. Run `hetu-online build-data --help`
for the full flag list.

### `hetu-online make-config`
Generates a LLaMA-Factory LoRA SFT YAML for one mode. Run once per mode
per task (`--mode cotgen`, `--mode cotcond`). Hyperparameters
(`--lora_rank`, `--cutoff_len`, `--per_device_train_batch_size`,
`--learning_rate`, `--num_train_epochs`, ...) are all overridable — see
`hetu-online make-config --help`.

### `hetu-online train`
`hetu-online train --model_family <qwen3|gemma4> <dataset_name> <cotgen|cotcond> <config.yaml> [-- extra llamafactory-cli args]`

`--model_family` must come before the positional args (see `--help`).
Launches `llamafactory-cli train` with `PYTHONPATH`,
`HETU_THINK_CONTENT_MASK`, and `HETU_THINK_OPEN_TAG`/`HETU_THINK_CLOSE_TAG`
all wired up correctly for the mode and family.

---

## How the masking actually works

`src/hetu_online/sitecustomize.py` is auto-imported by Python at
interpreter startup once its directory is on `PYTHONPATH` (which
`hetu-online train` sets for you). It monkeypatches
`llamafactory.data.processor.supervised.SupervisedDatasetProcessor._encode_data_example`.
After LLaMA-Factory encodes an example normally (full loss on the whole
response), the patch:

1. Finds where the response span starts in the token sequence (first
   non-masked label).
2. **Decodes** that span back to text (not a raw token-ID search — see
   "Why decode, not token IDs" below) and finds `<think>` / `</think>`
   by string search, scoped to the response span only.
3. Maps those character offsets back to token indices.
4. Sets `IGNORE_INDEX` on every label token strictly between the tags.
   Tag tokens and everything after `</think>` (the answer) are left alone.
5. If either tag is missing or malformed (e.g. truncated by `cutoff_len`),
   it **fails open** — leaves the example fully supervised rather than
   crashing or silently zeroing the whole example's loss.

### Why decode, not token IDs

An earlier, simpler version of this patch searched for the raw token-ID
sequence of `<think>`/`</think>` inside `input_ids`. That only works when
a tokenizer treats those strings as dedicated, atomic special tokens in
*every* context (true for Qwen3). It silently breaks on tokenizers where
the tags are plain BPE text — confirmed on GLM-Z1-9B, where `</think>`
tokenized standalone differs from `</think>` embedded in surrounding text,
because BPE merges are context-dependent. A token-ID search there finds
nothing, and the old code's fallback was "mask nothing" — i.e. it silently
trained full-loss on every example for that model, no error raised.
Decoding first and mapping character offsets back to token indices is
immune to this: it reconstructs the same text regardless of where the
tokenizer happens to draw subword boundaries. This is why the masking
mechanism itself needed no changes to support Gemma-4 after Qwen3 — but
tokenizer-agnostic masking logic is a necessary condition for supporting a
new family, not a sufficient one; see "Model families" below for what
else has to be verified first.

If your task's prompt text happens to contain the literal string
`<think>` (e.g. an instruction that says "reason inside `<think></think>`
tags"), note that the search is scoped to the **response** span only —
it will not accidentally match that mention in the prompt.

---

## Extending to a new dataset

This tool is dataset-agnostic within the two supported model families —
nothing in the data pipeline is specific to any one task. To point it at a
new dataset:

1. Get your raw data into the `{prompt, thinking, answer}` shape (write a
   small one-off script if your source format differs — that's the only
   per-dataset code you should ever need to write).
2. Run `hetu-online build-data` with a new `--dataset_name` and the
   `--model_family` you're training.
3. Run `hetu-online make-config` twice (cotgen, cotcond) with your base
   model (any size within that family) and desired hyperparameters.
4. `hetu-online train --model_family <family> <dataset_name> <mode> <config>`.

Nothing about steps 2-4 changes between tasks. Multiple datasets can
share one `LLaMA-Factory/data` dir and one `dataset_info.json` — each
`--dataset_name` gets its own namespaced keys (`<name>_cotgen_train`,
`<name>_cotcond_val`, etc.), so they won't collide.

### Model families: what's supported and why it's closed

Only `qwen3` and `gemma4` are accepted by `--model_family` — this is
deliberate, not a current limitation waiting to be lifted by passing a
different string. `model_families.py` is the single source of truth for
each family's template name, think-open/think-close tags,
freeze_vision_tower/freeze_multi_modal_projector, flash_attn,
enable_liger_kernel, and default cutoff_len; every fact in it is cited
back to its verification source (HF Hub chat templates, LLaMA-Factory's
own registry source, or a real measured OOM). The masking patch itself
(`sitecustomize.py`) is decode-based and therefore tokenizer-agnostic in
principle, but that's not the same as *supported*: a new family's think
tags or template name failing to match reality fails **silently**
(CotCond masking fails open to full supervision, no error, no crash), so
adding one requires the same level of citation-backed verification as the
existing two, not just a quick edit.

### Gemma-4: code shipped, not yet CLI-enabled

Gemma-4 (E2B/E4B/12B) support is fully written — `model_families.py` has
a cited, verified `FamilyConfig` for it, `config_builder.py` emits the
right YAML keys for it, and there's a vendored LLaMA-Factory patch it
needs (below) — but `cli.py`'s `CLI_ENABLED_MODEL_FAMILIES` currently
restricts `--model_family` to `("qwen3",)` only, because no Gemma-4
training run has completed successfully end-to-end on real GPU hardware
yet. Re-enabling it once that validation happens (on a GPU machine) is a
one-line change: flip `CLI_ENABLED_MODEL_FAMILIES` back to
`MODEL_FAMILIES` in `cli.py` — no other code changes needed. Until then,
`hetu-online build-data|make-config|train --model_family gemma4` is
refused by argparse with a clear error, not silently accepted.

If you're doing that validation work, Gemma-4 needs one thing beyond a
pip install that Qwen3 doesn't: LLaMA-Factory's liger-kernel dispatcher is
missing a `gemma4`/`gemma4_text` branch as of the pinned upstream commit
this tool was built against, so `enable_liger_kernel: true` (which
`make-config` will set automatically once gemma4 is CLI-enabled) silently
does nothing without it — and without liger's fused linear
cross-entropy, Gemma-4's 262144-vocab logits tensor is large enough to
OOM at the validated `cutoff_len`. Apply the vendored patch to your
LLaMA-Factory checkout once, in the same environment `llamafactory-cli`
runs in:

```bash
cd /path/to/LLaMA-Factory
git apply /path/to/HETU_Online/src/hetu_online/patches/llamafactory_gemma4_liger.patch
```

Neither `make-config` nor `train` can detect whether the patch is
actually applied — LLaMA-Factory silently keeps running (just without the
liger speedup/memory savings) if you skip this step, so at long context
you may not notice until you OOM.

---

## Running training + logging a run (`run-one`, `summarize-training`)

`hetu-online run-one` wraps `hetu-online train` with the operational
pieces a real sweep needs: per-run log file, GPU-memory polling (samples
`nvidia-smi` on a background thread every 5s, records the peak), automatic
pruning of all but the newest checkpoint on success (saves disk), and a
`{dataset_name}_{mode}_summary.json` (status, wall-clock time, peak GPU
memory, and LLaMA-Factory's own `train_results.json` if present) written
to `--results_dir`.

```bash
hetu-online run-one --dataset_name my_task --mode cotgen --model_family qwen3 \
  --config configs/my_task_cotgen.yaml --gpu_ids 0 \
  --results_dir results --logs_dir logs
```

`--gpu_ids` accepts a comma-separated list (e.g. `0,1`) for multi-GPU —
same as setting `CUDA_VISIBLE_DEVICES` yourself before `hetu-online
train`, just also included in the summary JSON.

Once you've run a batch of these, `hetu-online summarize-training
--results_dir results` joins every `*_summary.json` there into one
`comparison.csv` (one row per run) for comparing CotGen vs CotCond
training cost/behavior across a sweep.

---

## Evaluating checkpoints (`eval`, `build-eval-table`, `build-pivot-table`)

**Needs the eval environment** (`pip install -e ".[eval]"` — see Install).
`hetu-online eval` loads a base model + LoRA adapter into vLLM, runs one
benchmark `--num_repeats` times (default 3; sampling is stochastic per the
model's own HF-card-recommended params, so repeats differ), grades every
response, and writes a full per-query JSONL trace plus a summary JSON with
per-repeat and aggregate accuracy, throughput, and reasoning-length stats.

```bash
hetu-online eval --checkpoint_dir checkpoints/my_task/cotgen/checkpoint-400 \
  --model_family qwen3 --dataset_name my_task --mode cotgen \
  --benchmark aime24 --output_dir eval_results
```

Pass `--mode base --checkpoint_dir` omitted to evaluate the un-fine-tuned
base model directly, for comparison against the two trained modes.
Benchmarks: `aime24`, `aime25`, `math500`, `gpqa_diamond` — all four are
pulled fresh from the Hugging Face Hub via `datasets.load_dataset`, so
there's no benchmark data to migrate between machines.

Eval's own `--model_family` choices are a **separate, wider** vocabulary
than training's (`hetu_online.model_families`): `qwen3`, `qwen3_0p6b`,
`qwen3_4b`, plus two DeepSeek-R1-Distill families (`dsr1_llama8b`,
`dsr1_qwen7b`) evaluated here for comparison even though they weren't
trained via this tool's CotGen/CotCond pipeline. See
`hetu_online/eval/eval_model_families.py`'s docstring before assuming the
two modules' `model_family` strings mean the same thing.

A generation that hits `--max_new_tokens` before naturally finishing is
**continued** (re-prompted with what it generated so far, given a fresh
budget), not truncated-and-scored-as-is or discarded — see
`hetu_online/eval/run_eval.py`'s module docstring for why, and for the
per-family continuation-round budgets that keep this within each model's
native context window.

Two table builders read the JSON summaries straight off disk (always
current, including a still-running eval's partial results):

- `hetu-online build-eval-table --root <project_root>` joins training +
  eval summaries into one wide CSV (`eval_results/final_comparison_table.csv`
  by default).
- `hetu-online build-pivot-table --root <project_root>` prints a
  base/cotgen/cotcond accuracy comparison matrix (requires pandas;
  `--csv` also writes it out).

Both default to `--root` = current directory, expecting `results/` and
`eval_results/` (matching what `run-one`/`eval` write by default) —
override with `--train_results_dir`/`--eval_results_dir` for a different
layout.

---

## Related approaches this tool does *not* implement

If you also want the "hint-in-prompt, bare-answer-only response" variant
(sometimes called plain CotCond elsewhere) — where the teacher's reasoning
is pasted into the *prompt* as a worked example and the model's own
response is just the bare answer, never its own `<think>` block — that's
a different objective (`P(answer | prompt, GIVEN reasoning)` rather than
`P(answer | prompt, SELF-GENERATED reasoning)`) and isn't what this tool
builds. It's a legitimate complementary setup (useful for testing whether
a model can *use* reasoning it's handed, as opposed to whether it can
*produce* useful reasoning on its own) — worth adding as a second data
builder here later if needed, but out of scope for this pass.

---

## Project layout

```
HETU_Online/
  pyproject.toml               # pip install -e .            -> trainer env, the `hetu-online` command
                                # pip install -e ".[eval]"    -> adds the eval env's deps
  src/hetu_online/
    cli.py                     # registers every `hetu-online <subcommand>`; CLI_ENABLED_MODEL_FAMILIES
                                #   is the training-side CLI gate (currently qwen3-only, see above)
    model_families.py          # qwen3/gemma4 template, tags, freeze/flash_attn/liger, cutoff_len
    data_builder.py            # prompt/thinking/answer -> ShareGPT (both modes)
    config_builder.py          # LLaMA-Factory YAML generator
    train.py                   # launches llamafactory-cli with PYTHONPATH + env vars set
    sitecustomize.py           # the masking mechanism itself (see above)
    patches/
      llamafactory_gemma4_liger.patch   # required LLaMA-Factory patch once --model_family gemma4 is CLI-enabled
    orchestration/
      run_one.py                # `hetu-online run-one` -- train + log + GPU-mem-poll + summary JSON
      summarize.py               # `hetu-online summarize-training` -- joins run-one summaries into one CSV
    eval/
      benchmarks.py              # AIME24/25, MATH500, GPQA-Diamond loaders + graders (pulled from HF Hub)
      eval_model_families.py     # eval's OWN (wider) model-family vocabulary -- see note above
      run_eval.py                # `hetu-online eval` -- vLLM generation + grading + continuation logic
      tables.py                  # `hetu-online build-eval-table`/`build-pivot-table`
  tests/                        # unittest suite -- run with:
                                 #   PYTHONPATH=src python -m unittest discover -s tests
```
