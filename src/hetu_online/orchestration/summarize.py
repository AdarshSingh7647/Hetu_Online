"""
Reads every {results_dir}/{dataset_name}_{mode}_summary.json produced by
`hetu-online run-one` and writes one comparison CSV with a row per run, for
comparing CotGen vs CotCond on training cost/behavior.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

FIELDS = [
    "dataset_name", "mode", "status", "gpu_ids", "num_gpus",
    "start_time", "end_time", "wall_clock_seconds",
    "peak_gpu_memory_mib_total",
    "train_runtime", "train_samples_per_second", "train_steps_per_second",
    "train_loss", "total_flos",
    "eval_loss",
    "output_dir", "config",
]


def flatten(summary: dict) -> dict:
    row = {k: summary.get(k) for k in FIELDS if k in summary}
    tr = summary.get("train_results") or {}
    for key in ("train_runtime", "train_samples_per_second",
                "train_steps_per_second", "train_loss", "total_flos", "eval_loss"):
        row[key] = tr.get(key)
    for k in FIELDS:
        row.setdefault(k, None)
    return row


def summarize_training(results_dir: str, out_csv: str) -> int:
    """Writes out_csv and returns the number of rows written (0 if no
    summary JSON files were found yet)."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*_summary.json")))
    rows = []
    for p in paths:
        with open(p) as f:
            summary = json.load(f)
        rows.append(flatten(summary))

    if not rows:
        print("[summarize] no summary JSON files found yet.")
        return 0

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summarize] wrote {len(rows)} rows to {out_csv}")
    return len(rows)


def _add_summarize_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "summarize-training",
        help="Join every results/*_summary.json (from `hetu-online run-one`) into one CSV.",
    )
    p.add_argument("--results_dir", default=os.path.join(os.getcwd(), "results"))
    p.add_argument("--out_csv", default=None, help="Default: <results_dir>/comparison.csv.")
    p.set_defaults(func=_run_summarize)


def _run_summarize(args: argparse.Namespace) -> int:
    out_csv = args.out_csv or os.path.join(args.results_dir, "comparison.csv")
    summarize_training(args.results_dir, out_csv)
    return 0
