"""
Builds comparison tables from the JSON summaries written by
hetu_online.orchestration.run_one (results/*_summary.json, training side)
and hetu_online.eval.run_eval (eval_results/*_summary.json, eval side).

Two views, both reading directly off disk each time (never cached), so
either reflects whatever repeats/runs have actually landed so far --
including a run still mid-flight, since run_eval writes its summary after
EVERY repeat, not just once at the end:

  final  -- one wide CSV row per (dataset_name, mode, benchmark), joining
            training-side columns (FLOPs, runtime, tokens/sec) onto each of
            that model/mode's benchmark rows.
  pivot  -- one row per (model, ablation, benchmark), with base_acc/
            cotgen_acc/cotcond_acc columns side by side for direct
            comparison, printed as a table and optionally also written to
            CSV. Requires pandas (only this view does).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

FINAL_TABLE_FIELDS = [
    "model",
    "ablation",
    "training_method",
    "benchmark",
    "run_id",
    "num_eval_repeats",
    "num_eval_repeats_completed",
    "eval_is_complete",
    "train_total_flos",
    "train_runtime_seconds",
    "train_num_input_tokens_seen",
    "train_tokens_per_sec",
    "eval_accuracy_mean",
    "eval_accuracy_std",
    "eval_accuracy_mean_excl_extended",
    "eval_accuracy_std_excl_extended",
    "eval_num_extended_total",
    "eval_num_extended_correct_total",
    "eval_num_extended_incorrect_total",
    "eval_extended_accuracy_mean",
    "eval_num_still_truncated_total",
    "eval_mean_extended_reasoning_tokens_added",
    "eval_output_tokens_per_sec_mean",
    "eval_input_tokens_per_sec_mean",
    "eval_queries_per_sec_mean",
    "eval_mean_reasoning_tokens",
    "eval_num_questions",
]

PIVOT_BENCHMARKS = ["aime24", "aime25", "math500", "gpqa_diamond"]
PIVOT_ABLATIONS = ["all", "correct_only"]
PIVOT_MODES = ["base", "cotgen", "cotcond"]


def parse_dataset_name(name: str) -> tuple[str, str]:
    """s1k11_all_qwen3 -> (qwen3, all); s1k11_correct_only_dsr1_llama8b ->
    (dsr1_llama8b, correct_only); base_qwen3 -> (qwen3, "base") for the
    un-fine-tuned base-model comparison rows. The "s1k11_" prefix here
    reflects this project's own dataset naming convention (simplescaling/
    s1K-1.1) -- a different dataset's --dataset_name won't parse against
    this and will raise ValueError, which is expected: this table format
    is specific to that naming scheme, not a general one."""
    if name.startswith("base_"):
        return name[len("base_"):], "base"
    if not name.startswith("s1k11_"):
        raise ValueError(f"unrecognized dataset_name: {name}")
    rest = name[len("s1k11_"):]
    if rest.startswith("correct_only_"):
        return rest[len("correct_only_"):], "correct_only"
    if rest.startswith("all_"):
        return rest[len("all_"):], "all"
    raise ValueError(f"unrecognized dataset_name: {name}")


def _load_train_summaries(train_results_dir: str) -> dict[tuple[str, str], dict]:
    out = {}
    for path in glob.glob(os.path.join(train_results_dir, "*_summary.json")):
        with open(path) as f:
            s = json.load(f)
        out[(s["dataset_name"], s["mode"])] = s
    return out


def build_final_table(train_results_dir: str, eval_results_dirs: list[str], out_csv: str) -> int:
    """Writes out_csv and returns the number of rows written (0 if no eval
    summaries were found yet)."""
    train_summaries = _load_train_summaries(train_results_dir)
    eval_paths = sorted(
        path
        for d in eval_results_dirs
        for path in glob.glob(os.path.join(d, "*_summary.json"))
    )

    if not eval_paths:
        print("[tables] no eval summary JSON files found yet.")
        return 0

    rows = []
    for path in eval_paths:
        with open(path) as f:
            e = json.load(f)

        dataset_name = e["dataset_name"]
        mode = e["mode"]
        model, ablation = parse_dataset_name(dataset_name)

        t = train_summaries.get((dataset_name, mode), {})
        tr = t.get("train_results") or {}

        train_runtime = tr.get("train_runtime")
        tokens_seen = tr.get("num_input_tokens_seen")
        train_tps = (tokens_seen / train_runtime) if (tokens_seen and train_runtime) else None

        rows.append(
            {
                "model": model,
                "ablation": ablation,
                "training_method": mode,
                "benchmark": e["benchmark"],
                "run_id": e.get("run_id"),
                "num_eval_repeats": e["num_repeats"],
                # Older summary files (written before incremental per-repeat
                # saving existed) never have these keys -- they were only ever
                # written once fully done, so default to "complete" for them
                # rather than misreporting a finished legacy run as partial.
                "num_eval_repeats_completed": e.get("num_repeats_completed", e["num_repeats"]),
                "eval_is_complete": e.get("is_complete", True),
                "train_total_flos": tr.get("total_flos"),
                "train_runtime_seconds": train_runtime,
                "train_num_input_tokens_seen": tokens_seen,
                "train_tokens_per_sec": train_tps,
                "eval_accuracy_mean": e["accuracy_mean"],
                "eval_accuracy_std": e["accuracy_std"],
                "eval_accuracy_mean_excl_extended": e.get("accuracy_mean_excl_extended"),
                "eval_accuracy_std_excl_extended": e.get("accuracy_std_excl_extended"),
                "eval_num_extended_total": e.get("num_extended_total"),
                "eval_num_extended_correct_total": e.get("num_extended_correct_total"),
                "eval_num_extended_incorrect_total": e.get("num_extended_incorrect_total"),
                "eval_extended_accuracy_mean": e.get("extended_accuracy_mean"),
                "eval_num_still_truncated_total": e.get("num_still_truncated_total"),
                "eval_mean_extended_reasoning_tokens_added": e.get("mean_extended_reasoning_tokens_added_mean"),
                "eval_output_tokens_per_sec_mean": e["output_tokens_per_sec_mean"],
                "eval_input_tokens_per_sec_mean": e["input_tokens_per_sec_mean"],
                "eval_queries_per_sec_mean": e["queries_per_sec_mean"],
                "eval_mean_reasoning_tokens": e["mean_reasoning_tokens_mean"],
                "eval_num_questions": e["num_questions"],
            }
        )

    rows.sort(
        key=lambda r: (
            r["model"],
            r["ablation"],
            r["training_method"],
            r["benchmark"],
            r["run_id"] is not None,
            r["run_id"] or "",
        )
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[tables] wrote {len(rows)} rows to {out_csv}")
    # Base-model rows (ablation == "base") never have a training summary --
    # there's no fine-tuning run to match, that's expected, not a warning.
    missing_train = [r for r in rows if r["train_total_flos"] is None and r["ablation"] != "base"]
    if missing_train:
        print(f"[tables] WARNING: {len(missing_train)} rows have no matching train summary")
    return len(rows)


def _load_all_eval_summaries(eval_results_dirs: list[str]):
    """One row per summary JSON file found (i.e. per run_id, not per cell --
    a cell with 2 separate run_ids contributes 2 rows here). Always reads
    the CURRENT file contents, so a still-running job's summary shows up
    with whatever num_repeats_completed it has reached so far."""
    import pandas as pd

    records = []
    for d in eval_results_dirs:
        for path in glob.glob(os.path.join(d, "*_summary.json")):
            with open(path) as f:
                e = json.load(f)
            model, ablation = parse_dataset_name(e["dataset_name"])
            records.append(
                {
                    "model": model,
                    "ablation": ablation,
                    "benchmark": e["benchmark"],
                    "mode": e["mode"],
                    "run_id": e.get("run_id"),
                    "accuracy_mean": e["accuracy_mean"],
                    "num_repeats_completed": e.get("num_repeats_completed", e["num_repeats"]),
                    "num_repeats_target": e["num_repeats"],
                    "is_complete": e.get("is_complete", True),
                    "source_path": path,
                }
            )
    return pd.DataFrame.from_records(records)


_EVAL_SUMMARY_COLUMNS = [
    "model", "ablation", "benchmark", "mode", "run_id", "accuracy_mean",
    "num_repeats_completed", "num_repeats_target", "is_complete", "source_path",
]


def _pick_best_run_per_cell(df):
    """When a (model, ablation, benchmark, mode) cell has multiple run_ids,
    keep only the one with the most completed repeats as the representative
    value for that cell. Ties keep the first encountered."""
    if df.empty:
        # An empty DataFrame built with no records has NO columns at all
        # (pandas can't infer them from zero rows), so .groupby(["model",
        # ...]) below would raise KeyError -- return an empty frame that at
        # least has the expected columns instead.
        import pandas as pd

        return pd.DataFrame(columns=_EVAL_SUMMARY_COLUMNS)
    idx = df.groupby(["model", "ablation", "benchmark", "mode"])["num_repeats_completed"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_pivot_table(eval_results_dirs: list[str], models: list[str]):
    """Returns a pandas DataFrame with one row per (model, ablation,
    benchmark) -- EVERY expected combination in models x PIVOT_ABLATIONS x
    PIVOT_BENCHMARKS, even ones with zero data on disk -- with base_acc,
    cotgen_acc, cotcond_acc, delta_cotcond_vs_cotgen_pct columns. Base mode
    has no ablation axis of its own, so the same base-model row is joined
    into BOTH the "all" and "correct_only" groups for a given model."""
    import pandas as pd

    df = _load_all_eval_summaries(eval_results_dirs)
    best = _pick_best_run_per_cell(df)

    base_rows = best[best["mode"] == "base"].copy() if not best.empty else best
    non_base_rows = best[best["mode"] != "base"].copy() if not best.empty else best

    expanded_base = []
    for ablation in PIVOT_ABLATIONS:
        dup = base_rows.copy()
        dup["ablation"] = ablation
        expanded_base.append(dup)
    best_expanded = (
        pd.concat([non_base_rows] + expanded_base, ignore_index=True) if not base_rows.empty else non_base_rows
    )

    full_index = pd.MultiIndex.from_product(
        [models, PIVOT_ABLATIONS, PIVOT_BENCHMARKS], names=["model", "ablation", "benchmark"]
    )
    skeleton = pd.DataFrame(index=full_index).reset_index()

    out_rows = []
    for _, row in skeleton.iterrows():
        cell = best_expanded[
            (best_expanded["model"] == row["model"])
            & (best_expanded["ablation"] == row["ablation"])
            & (best_expanded["benchmark"] == row["benchmark"])
        ]
        entry = {"model": row["model"], "ablation": row["ablation"], "benchmark": row["benchmark"]}
        by_mode = {r["mode"]: r for _, r in cell.iterrows()}
        for mode in PIVOT_MODES:
            r = by_mode.get(mode)
            entry[f"{mode}_acc"] = r["accuracy_mean"] if r is not None else None
            entry[f"{mode}_progress"] = f"{r['num_repeats_completed']}/{r['num_repeats_target']}" if r is not None else None
        if entry["cotgen_acc"] is not None and entry["cotcond_acc"] is not None and entry["cotgen_acc"] != 0:
            entry["delta_cotcond_vs_cotgen_pct"] = (
                (entry["cotcond_acc"] - entry["cotgen_acc"]) / entry["cotgen_acc"] * 100
            )
        else:
            entry["delta_cotcond_vs_cotgen_pct"] = None
        out_rows.append(entry)

    return pd.DataFrame(out_rows)


def _default_eval_dirs(root: str) -> list[str]:
    return [os.path.join(root, "eval_results"), os.path.join(root, "eval_results_notebook")]


def _add_build_final_table_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-eval-table",
        help="Join training + eval summaries into one wide comparison CSV.",
    )
    p.add_argument("--root", default=os.getcwd(), help="Project root containing results/ and eval_results/.")
    p.add_argument("--train_results_dir", default=None, help="Default: <root>/results.")
    p.add_argument("--eval_results_dir", action="append", default=None,
                    help="Repeatable. Default: <root>/eval_results and <root>/eval_results_notebook.")
    p.add_argument("--out_csv", default=None, help="Default: <root>/eval_results/final_comparison_table.csv.")
    p.set_defaults(func=_run_build_final_table)


def _run_build_final_table(args: argparse.Namespace) -> int:
    train_results_dir = args.train_results_dir or os.path.join(args.root, "results")
    eval_results_dirs = args.eval_results_dir or _default_eval_dirs(args.root)
    out_csv = args.out_csv or os.path.join(args.root, "eval_results", "final_comparison_table.csv")
    build_final_table(train_results_dir, eval_results_dirs, out_csv)
    return 0


def _add_build_pivot_table_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-pivot-table",
        help="Print a base/cotgen/cotcond accuracy comparison matrix (requires pandas).",
    )
    p.add_argument("--root", default=os.getcwd(), help="Project root containing eval_results/.")
    p.add_argument("--eval_results_dir", action="append", default=None,
                    help="Repeatable. Default: <root>/eval_results and <root>/eval_results_notebook.")
    p.add_argument(
        "--model", action="append", dest="models", default=None,
        help="Repeatable. Which model_family rows to include. Default: qwen3, qwen3_0p6b, qwen3_4b, "
        "dsr1_llama8b, dsr1_qwen7b (eval's full supported family list).",
    )
    p.add_argument("--csv", default=None, help="Optional path to also write the pivot table as CSV.")
    p.set_defaults(func=_run_build_pivot_table)


def _run_build_pivot_table(args: argparse.Namespace) -> int:
    from .eval_model_families import MODEL_FAMILIES

    eval_results_dirs = args.eval_results_dir or _default_eval_dirs(args.root)
    models = args.models or list(MODEL_FAMILIES)
    pivot = build_pivot_table(eval_results_dirs, models)

    import pandas as pd

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    print(pivot.to_string(index=False))

    if args.csv:
        pivot.to_csv(args.csv, index=False)
        print(f"\nwrote {len(pivot)} rows to {args.csv}")
    return 0


def register_table_subcommands(subparsers) -> None:
    _add_build_final_table_parser(subparsers)
    _add_build_pivot_table_parser(subparsers)
