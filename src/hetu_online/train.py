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


def _patches_dir() -> str:
    """Directory containing sitecustomize.py, regardless of how this
    package was installed (editable install, wheel, or a plain checkout)."""
    return str(Path(__file__).resolve().parent)


def run_training(
    dataset_name: str,
    mode: str,
    config_path: str,
    extra_args: Optional[List[str]] = None,
) -> int:
    if mode not in ("cotgen", "cotcond"):
        raise ValueError(f"mode must be 'cotgen' or 'cotcond', got {mode!r}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")

    if shutil.which("llamafactory-cli") is None:
        raise RuntimeError(
            "llamafactory-cli not found on PATH. Install LLaMA-Factory first "
            "(pip install -e /path/to/LLaMA-Factory) and activate the same "
            "environment before running `hetu-online train`."
        )

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

    print(f"[hetu-online] config={config_path}")
    print(f"[hetu-online] PYTHONPATH={env['PYTHONPATH']}")

    cmd = ["llamafactory-cli", "train", config_path] + (extra_args or [])
    result = subprocess.run(cmd, env=env)
    return result.returncode
