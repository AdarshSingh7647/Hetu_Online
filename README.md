# HETU_Online

A small, pip-installable tool for training a model two ways on the same
reasoning data — **CotGen** and **CotCond** — on top of
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Give it a base
model and a JSON file of `{prompt, thinking, answer}` examples, and it
produces ready-to-run LoRA SFT configs for both modes, plus the training-time
loss masking CotCond needs (which stock LLaMA-Factory can't do on its own).

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

```bash
git clone <this-repo-url> HETU_Online
cd HETU_Online
pip install -e .

# LLaMA-Factory is the training backend and is a separate install --
# do this once, in the same environment:
pip install -e /path/to/LLaMA-Factory
```

This gives you the `hetu-online` command from anywhere — no need to `cd`
into the repo or remember script order.

---

## Quickstart

```bash
# 1. Build both datasets from one raw file (prompt/thinking/answer JSON)
hetu-online build-data \
  --input /path/to/raw_train.json \
  --out_dir /path/to/LLaMA-Factory/data/my_task \
  --dataset_name my_task \
  --llamafactory_data_dir /path/to/LLaMA-Factory/data \
  --system_prompt "You are a careful, helpful assistant."

# 2. Generate a config for each mode
hetu-online make-config --dataset_name my_task --mode cotgen  \
  --model_path /path/to/base_model --output_dir /path/to/checkpoints/my_task/cotgen \
  --template qwen3

hetu-online make-config --dataset_name my_task --mode cotcond \
  --model_path /path/to/base_model --output_dir /path/to/checkpoints/my_task/cotcond \
  --template qwen3

# 3. Train
hetu-online train my_task cotgen  configs/my_task_cotgen.yaml
hetu-online train my_task cotcond configs/my_task_cotcond.yaml
```

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
`hetu-online train <dataset_name> <cotgen|cotcond> <config.yaml> [-- extra llamafactory-cli args]`

Launches `llamafactory-cli train` with `PYTHONPATH` and
`HETU_THINK_CONTENT_MASK` wired up correctly for the mode.

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
tokenizer happens to draw subword boundaries, so it's correct for any
tokenizer, not just the one you first tested against.

If your task's prompt text happens to contain the literal string
`<think>` (e.g. an instruction that says "reason inside `<think></think>`
tags"), note that the search is scoped to the **response** span only —
it will not accidentally match that mention in the prompt.

---

## Extending to a new dataset

This tool is deliberately dataset-agnostic — nothing in it is specific to
any one task. To point it at a new dataset:

1. Get your raw data into the `{prompt, thinking, answer}` shape (write a
   small one-off script if your source format differs — that's the only
   per-dataset code you should ever need to write).
2. Run `hetu-online build-data` with a new `--dataset_name`.
3. Run `hetu-online make-config` twice (cotgen, cotcond) with your base
   model and desired hyperparameters.
4. `hetu-online train <dataset_name> <mode> <config>`.

Nothing about steps 2-4 changes between tasks. Multiple datasets can
share one `LLaMA-Factory/data` dir and one `dataset_info.json` — each
`--dataset_name` gets its own namespaced keys (`<name>_cotgen_train`,
`<name>_cotcond_val`, etc.), so they won't collide.

### Using a different chat template / tokenizer

Pass `--template <name>` to `make-config` (any LLaMA-Factory template:
`qwen3`, `glm4`, `llama3`, ...). The masking patch itself needs no changes
— it's decode-based and therefore tokenizer-agnostic (see above). If your
model uses different think-tag strings than `<think>`/`</think>`, set
`HETU_THINK_OPEN_TAG` / `HETU_THINK_CLOSE_TAG` in your shell before
running `hetu-online train` (rare enough that there's no dedicated flag
for it, but the patch reads them from the environment the same way).

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
  pyproject.toml               # pip install -e .  ->  the `hetu-online` command
  src/hetu_online/
    cli.py                     # `hetu-online build-data|make-config|train`
    data_builder.py            # prompt/thinking/answer -> ShareGPT (both modes)
    config_builder.py          # LLaMA-Factory YAML generator
    train.py                   # launches llamafactory-cli with PYTHONPATH + env var set
    sitecustomize.py           # the masking mechanism itself (see above)
```
