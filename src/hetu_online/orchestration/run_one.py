"""
Runs exactly one (dataset_name, mode) training job pinned to one or more
GPUs, with logging, GPU-memory polling, checkpoint pruning, and a summary
JSON -- the reusable unit a batch/sweep driver calls per cell.

This is a Python port of the original run_one.sh, using
hetu_online.train.run_training directly (in-process) instead of shelling
out to the `hetu-online` CLI, and a background thread instead of a forked
`nvidia-smi` polling loop for GPU-memory sampling.

Usage (CLI, via the package entrypoint):
  hetu-online run-one --dataset_name <name> --mode <cotgen|cotcond> \\
      --model_family <qwen3> --config <path/to/config.yaml> \\
      --gpu_ids 0[,1,...] --results_dir <dir> --logs_dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

from ..train import run_training


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GpuMemPoller:
    """Samples `nvidia-smi --query-gpu=memory.used` for the given GPU ids
    every `interval` seconds on a background thread, summed across all
    ids into one per-timestamp total, and writes rows to `csv_path` as it
    goes -- same shape as run_one.sh's polling subshell, just a thread
    instead of a forked loop so this stays a single Python process."""

    def __init__(self, gpu_ids: str, csv_path: str, interval: float = 5.0):
        self.gpu_ids = gpu_ids
        self.csv_path = csv_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mem_mib: float = 0.0

    def _sample_once(self) -> Optional[float]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", self.gpu_ids],
                capture_output=True, text=True, check=True,
            ).stdout
            total = sum(float(line.strip()) for line in out.splitlines() if line.strip())
            return total
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None

    def _run(self) -> None:
        with open(self.csv_path, "w") as f:
            f.write("timestamp,memory_used_mib_total\n")
            while not self._stop.is_set():
                mem = self._sample_once()
                if mem is not None:
                    f.write(f"{int(time.time())},{mem}\n")
                    f.flush()
                    self.peak_mem_mib = max(self.peak_mem_mib, mem)
                self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)


def _prune_checkpoints(out_dir: str) -> None:
    """Keeps only the highest-numbered checkpoint-N dir, deletes the rest --
    matches run_one.sh's disk-space-saving behavior after a successful run."""
    if not os.path.isdir(out_dir):
        return
    checkpoint_dirs = glob.glob(os.path.join(out_dir, "checkpoint-*"))
    best, best_n = None, -1
    for d in checkpoint_dirs:
        m = re.search(r"checkpoint-(\d+)$", d)
        if m and int(m.group(1)) > best_n:
            best_n, best = int(m.group(1)), d
    if best is None:
        return
    for d in checkpoint_dirs:
        if d != best:
            shutil.rmtree(d, ignore_errors=True)


def run_one(
    dataset_name: str,
    mode: str,
    model_family: str,
    config_path: str,
    gpu_ids: str,
    results_dir: str,
    logs_dir: str,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Runs one training job, writes {logs_dir}/{dataset_name}_{mode}.log,
    {logs_dir}/{dataset_name}_{mode}_gpumem.csv, and
    {results_dir}/{dataset_name}_{mode}_summary.json. Returns the training
    process's exit code (0 = success)."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    run_all_log = os.path.join(logs_dir, "run_all.log")
    run_log_path = os.path.join(logs_dir, f"{dataset_name}_{mode}.log")
    gpumem_log_path = os.path.join(logs_dir, f"{dataset_name}_{mode}_gpumem.csv")
    summary_path = os.path.join(results_dir, f"{dataset_name}_{mode}_summary.json")

    def log_line(msg: str) -> None:
        line = f"[{_now_iso()}] [run={dataset_name}_{mode} gpus={gpu_ids}] {msg}"
        print(line)
        with open(run_all_log, "a") as f:
            f.write(line + "\n")

    with open(config_path) as f:
        config_text = f.read()
    m = re.search(r"^output_dir:\s*(.+)$", config_text, re.MULTILINE)
    out_dir = m.group(1).strip() if m else None

    log_line(f"STARTING (config: {config_path})")
    start_ts = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    poller = GpuMemPoller(gpu_ids, gpumem_log_path)
    poller.start()

    exit_code = 1
    try:
        with open(run_log_path, "w") as run_log_f:
            import contextlib
            import io

            # run_training prints via print(), which writes to the real
            # stdout/stderr -- redirect both into the per-run log file, same
            # as run_one.sh's `> "$run_log" 2>&1`.
            with contextlib.redirect_stdout(run_log_f), contextlib.redirect_stderr(run_log_f):
                exit_code = run_training(
                    dataset_name=dataset_name,
                    mode=mode,
                    config_path=config_path,
                    model_family=model_family,
                    extra_args=extra_args,
                )
    finally:
        poller.stop()

    end_ts = time.time()
    elapsed = int(end_ts - start_ts)
    status = "SUCCESS" if exit_code == 0 else "FAILED"

    if exit_code == 0 and out_dir:
        _prune_checkpoints(out_dir)

    log_line(
        f"FINISHED -- {status} -- elapsed {elapsed // 3600}h{(elapsed % 3600) // 60}m{elapsed % 60}s "
        f"-- peak_gpu_mem_mib={poller.peak_mem_mib}"
    )

    train_results = None
    if out_dir:
        train_results_path = os.path.join(out_dir, "train_results.json")
        if os.path.isfile(train_results_path):
            with open(train_results_path) as f:
                train_results = json.load(f)

    summary = {
        "dataset_name": dataset_name,
        "mode": mode,
        "gpu_ids": gpu_ids,
        "num_gpus": len(gpu_ids.split(",")),
        "config": config_path,
        "output_dir": out_dir,
        "status": status,
        "start_time": datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_clock_seconds": elapsed,
        "peak_gpu_memory_mib_total": poller.peak_mem_mib or None,
        "train_results": train_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if exit_code != 0:
        log_line(f"FAILED (exit={exit_code}). See {run_log_path} for the full log.")
        try:
            with open(run_log_path) as f:
                tail = f.readlines()[-50:]
            log_line("Last 50 lines:\n" + "".join(tail))
        except OSError:
            pass

    return exit_code


def _add_run_one_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "run-one",
        help="Run one (dataset_name, mode) training job with logging, GPU-mem polling, and a summary JSON.",
    )
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--mode", required=True, choices=["cotgen", "cotcond"])
    p.add_argument("--model_family", required=True)
    p.add_argument("--config", required=True, dest="config_path", help="Path to the YAML config (from make-config).")
    p.add_argument("--gpu_ids", required=True, help="Comma-separated CUDA device ids, e.g. '0' or '0,1'.")
    p.add_argument("--results_dir", default=os.path.join(os.getcwd(), "results"))
    p.add_argument("--logs_dir", default=os.path.join(os.getcwd(), "logs"))
    p.add_argument("extra_args", nargs=argparse.REMAINDER,
                    help="Extra args passed through to llamafactory-cli train.")
    p.set_defaults(func=_run_run_one)


def _run_run_one(args: argparse.Namespace) -> int:
    return run_one(
        dataset_name=args.dataset_name,
        mode=args.mode,
        model_family=args.model_family,
        config_path=args.config_path,
        gpu_ids=args.gpu_ids,
        results_dir=args.results_dir,
        logs_dir=args.logs_dir,
        extra_args=args.extra_args,
    )
