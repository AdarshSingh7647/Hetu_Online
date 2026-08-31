"""
HETU_Online data builder -- turns raw (prompt, thinking, answer) examples
into LLaMA-Factory ShareGPT SFT data for BOTH training modes:

  cotgen   -- gpt turn = "<think>\\n{thinking}\\n</think>\\n\\n{answer}", full loss.
              Model imitates the teacher's reasoning AND answer directly.

  cotcond  -- gpt turn = "<think>\\n{thinking}\\n</think>\\n\\n{answer}", SAME
              text as cotgen, but loss is masked on the reasoning CONTENT
              at train time (via the sitecustomize patch + HETU_THINK_CONTENT_MASK=1).
              Only the <think>/</think> tag tokens and the answer get
              gradient. The model still learns to always open/close a
              think block and answer well GIVEN whatever reasoning it
              generates at inference -- but the reasoning's content is
              never directly supervised, so it stays shaped by the base
              model's own generation ability, not by imitation of this
              corpus's specific reasoning phrasing.

This is "Family B" (Math_Reasoning's cotcond + passage_reranking's
cotcond_think, from model-forge) -- the variant where the model generates
its own reasoning at both train and test time, as opposed to "Family A"
(teacher reasoning pasted into the prompt as a hint, bare answer as the
only response -- a different objective: P(answer | prompt, GIVEN
reasoning), not P(answer | prompt, SELF-GENERATED reasoning)).

cotgen and cotcond datasets are BYTE-IDENTICAL in this design -- the only
difference between the two training runs is whether HETU_THINK_CONTENT_MASK=1
is exported at launch. They are still written as separate registered
datasets/files (rather than one dataset reused twice) so a run's
`dataset:` key in its YAML config unambiguously records which mode
produced that checkpoint, and so cotgen/cotcond can point at genuinely
different data later if a task ever needs that (e.g. a hint prepended
only for one mode) without restructuring this tool.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

REQUIRED_FIELDS = ["prompt", "thinking", "answer"]
MODES = ("cotgen", "cotcond")


def load_examples(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    for i, ex in enumerate(data):
        missing = [k for k in REQUIRED_FIELDS if k not in ex or ex[k] is None]
        if missing:
            raise ValueError(
                f"Example {i} in {path} is missing required field(s) {missing}. "
                f"Every example needs: {REQUIRED_FIELDS}"
            )
    return data


def make_conversation(human: str, gpt: str, system: str) -> Dict[str, Any]:
    return {
        "system": system,
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt},
        ],
    }


DEFAULT_THINK_OPEN = "<think>\n"
DEFAULT_THINK_CLOSE = "\n</think>\n\n"


def build_think_response(
    thinking: str,
    answer: str,
    think_open: str = DEFAULT_THINK_OPEN,
    think_close: str = DEFAULT_THINK_CLOSE,
) -> str:
    """think_open/think_close default to Qwen3/DeepSeek's tags. A model
    family with different reasoning-span delimiters (e.g. Gemma-4's
    "<|channel>thought\\n" / "<channel|>") must pass its own pair here --
    these MUST match whatever HETU_THINK_OPEN_TAG/HETU_THINK_CLOSE_TAG are
    set to at train time (see sitecustomize.py), or CotCond's masking will
    silently fail to find the tags and fail open to full supervision with
    no error."""
    thinking = thinking.strip()
    answer = answer.strip()
    return f"{think_open}{thinking}{think_close}{answer}"


def build_examples(
    raw: List[Dict[str, Any]],
    system_prompt: str,
    think_open: str = DEFAULT_THINK_OPEN,
    think_close: str = DEFAULT_THINK_CLOSE,
) -> List[Dict[str, Any]]:
    """cotgen and cotcond share this exact same shape -- see module
    docstring for why the distinction lives entirely in the training-time
    loss mask, not in the data."""
    out = []
    for ex in raw:
        response = build_think_response(ex["thinking"], ex["answer"], think_open, think_close)
        out.append(make_conversation(ex["prompt"], response, system_prompt))
    return out


def write_json(data: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def split_val(
    data: List[Dict[str, Any]], val_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_fraction <= 0:
        return data, []
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n_val = max(1, int(len(data) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [d for i, d in enumerate(data) if i not in val_idx]
    val = [d for i, d in enumerate(data) if i in val_idx]
    return train, val


def register_datasets(llamafactory_data_dir: str, entries: Dict[str, str]) -> None:
    """Adds/updates entries in LLaMA-Factory's data/dataset_info.json so
    `dataset: <key>` in a training YAML resolves to the right file. Merges
    into the existing file rather than overwriting it -- safe to run
    against a dataset_info.json that already has other tasks registered."""
    info_path = os.path.join(llamafactory_data_dir, "dataset_info.json")
    info: Dict[str, Any] = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)

    for key, abs_path in entries.items():
        try:
            file_name = os.path.relpath(abs_path, llamafactory_data_dir)
            if file_name.startswith(".."):
                file_name = abs_path
        except ValueError:
            file_name = abs_path

        info[key] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt",
            },
        }

    os.makedirs(llamafactory_data_dir, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[hetu-online] registered {len(entries)} dataset(s) in {info_path}")


def build_cot_data(
    input_path: str,
    out_dir: str,
    dataset_name: str,
    llamafactory_data_dir: str,
    val_input: str = None,
    val_fraction: float = 0.02,
    system_prompt: str = "You are a careful, helpful assistant.",
    seed: int = 12345,
    think_open: str = DEFAULT_THINK_OPEN,
    think_close: str = DEFAULT_THINK_CLOSE,
) -> Dict[str, str]:
    """Core entry point (importable, used by the CLI). Returns the dict of
    registered dataset key -> written file path."""
    train_raw = load_examples(input_path)
    if val_input:
        val_raw = load_examples(val_input)
    else:
        train_raw, val_raw = split_val(train_raw, val_fraction, seed)

    train_examples = build_examples(train_raw, system_prompt, think_open, think_close)
    val_examples = build_examples(val_raw, system_prompt, think_open, think_close)

    written: Dict[str, str] = {}
    for mode in MODES:
        train_path = os.path.join(out_dir, f"{dataset_name}_{mode}_train.json")
        val_path = os.path.join(out_dir, f"{dataset_name}_{mode}_val.json")
        write_json(train_examples, train_path)
        write_json(val_examples, val_path)
        written[f"{dataset_name}_{mode}_train"] = train_path
        written[f"{dataset_name}_{mode}_val"] = val_path
        print(f"[hetu-online] wrote {mode}: train={len(train_examples)} rows -> {train_path}")
        print(f"[hetu-online] wrote {mode}: val={len(val_examples)} rows -> {val_path}")

    register_datasets(llamafactory_data_dir, written)
    return written
