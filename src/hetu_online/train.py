"""
Launches `llamafactory-cli train <config>` with the right PYTHONPATH and
env var for the given mode, so cotcond's masking requirement can never be
silently forgotten -- a bare `llamafactory-cli train cotcond.yaml` with no
env var will silently train FULL LOSS instead of think-content-masked
loss. No error, just the wrong objective. This wraps that launch so the
env var is set automatically from `mode`.

Multi-GPU: llamafactory-cli respects CUDA_VISIBLE_DEVICES / your own
accelerate config as usual -- set those in your shell before calling this
(or before `hetu-online train`), same as any other llamafactory-cli run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import model_families


def _patches_dir() -> str:
    """Directory containing sitecustomize.py, regardless of how this
    package was installed (editable install, wheel, or a plain checkout)."""
    return str(Path(__file__).resolve().parent)


def run_training(
    dataset_name: str,
    mode: str,
    config_path: str,
    model_family: str,
    extra_args: Optional[List[str]] = None,
) -> int:
    """model_family (see model_families.py) supplies the reasoning-span
    open/close tags for CotCond's masking patch -- Qwen3 uses
    "<think>"/"</think>", Gemma-4 uses "<|channel>thought\\n"/"<channel|>".
    These MUST match whatever the training data's think-block was actually
    written with (data_builder.py's build_cot_data, given the SAME
    model_family) or CotCond's masking silently fails open to full
    supervision with no error -- which is exactly why this takes
    model_family instead of raw tag strings: the two call sites can no
    longer disagree."""
    if mode not in ("cotgen", "cotcond"):
        raise ValueError(f"mode must be 'cotgen' or 'cotcond', got {mode!r}")

    think_open_tag, think_close_tag = model_families.think_tags(model_family)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")

    if shutil.which("llamafactory-cli") is None:
        raise RuntimeError(
            "llamafactory-cli not found on PATH. Install LLaMA-Factory first "
            "(pip install -e /path/to/LLaMA-Factory) and activate the same "
            "environment before running `hetu-online train`."
        )

    if model_family == "gemma4":
        print("[hetu-online] model_family=gemma4 requires LLaMA-Factory's gemma4 "
              "liger-kernel dispatch patch (see this package's "
              "patches/llamafactory_gemma4_liger.patch) -- without it, "
              "enable_liger_kernel: true in the config silently does nothing.")

    env = os.environ.copy()
    patches_dir = _patches_dir()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{patches_dir}:{existing_pp}" if existing_pp else patches_dir

    if mode == "cotcond":
        env["HETU_THINK_CONTENT_MASK"] = "1"
        print(f"[hetu-online] dataset={dataset_name} mode=cotcond -- "
              f"HETU_THINK_CONTENT_MASK=1 (reasoning content masked, tags+answer supervised)")
    else:
        env.pop("HETU_THINK_CONTENT_MASK", None)
        print(f"[hetu-online] dataset={dataset_name} mode=cotgen -- "
              f"full loss on reasoning+answer, no masking")

    env["HETU_THINK_OPEN_TAG"] = think_open_tag
    env["HETU_THINK_CLOSE_TAG"] = think_close_tag
    print(f"[hetu-online] model_family={model_family} "
          f"HETU_THINK_OPEN_TAG={think_open_tag!r} HETU_THINK_CLOSE_TAG={think_close_tag!r}")

    print(f"[hetu-online] config={config_path}")
    print(f"[hetu-online] PYTHONPATH={env['PYTHONPATH']}")

    cmd = ["llamafactory-cli", "train", config_path] + (extra_args or [])
    result = subprocess.run(cmd, env=env)
    return result.returncode
