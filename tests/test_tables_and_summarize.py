"""Tests for the ported orchestration/eval-table-building logic
(summarize.py, eval/tables.py) -- pure file/CSV logic, no GPU/vLLM
required, using synthetic summary JSON fixtures shaped like what
run_one.py / eval/run_eval.py actually write."""

import csv
import json
import os
import tempfile
import unittest

from hetu_online.eval import tables
from hetu_online.orchestration import summarize


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


class TestParseDatasetName(unittest.TestCase):
    def test_all_ablation(self):
        self.assertEqual(tables.parse_dataset_name("s1k11_all_qwen3"), ("qwen3", "all"))

    def test_correct_only_ablation(self):
        self.assertEqual(
            tables.parse_dataset_name("s1k11_correct_only_dsr1_llama8b"),
            ("dsr1_llama8b", "correct_only"),
        )

    def test_base(self):
        self.assertEqual(tables.parse_dataset_name("base_qwen3"), ("qwen3", "base"))

    def test_unrecognized_raises(self):
        with self.assertRaises(ValueError):
            tables.parse_dataset_name("totally_unrelated_name")


class TestSummarizeTraining(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_writes_one_row_per_summary(self):
        results_dir = os.path.join(self.tmpdir.name, "results")
        _write_json(
            os.path.join(results_dir, "s1k11_all_qwen3_cotgen_summary.json"),
            {
                "dataset_name": "s1k11_all_qwen3", "mode": "cotgen", "status": "SUCCESS",
                "gpu_ids": "0", "num_gpus": 1, "start_time": "t0", "end_time": "t1",
                "wall_clock_seconds": 100, "peak_gpu_memory_mib_total": 1234.0,
                "output_dir": "/ckpt", "config": "/cfg.yaml",
                "train_results": {"train_runtime": 90.0, "train_loss": 0.5, "total_flos": 42},
            },
        )
        out_csv = os.path.join(self.tmpdir.name, "comparison.csv")
        n = summarize.summarize_training(results_dir, out_csv)
        self.assertEqual(n, 1)
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_name"], "s1k11_all_qwen3")
        self.assertEqual(rows[0]["train_loss"], "0.5")

    def test_no_summaries_returns_zero(self):
        results_dir = os.path.join(self.tmpdir.name, "empty_results")
        os.makedirs(results_dir)
        n = summarize.summarize_training(results_dir, os.path.join(self.tmpdir.name, "out.csv"))
        self.assertEqual(n, 0)


def _eval_summary(dataset_name, mode, benchmark, accuracy_mean=0.5, run_id=None):
    return {
        "dataset_name": dataset_name, "mode": mode, "benchmark": benchmark,
        "run_id": run_id, "num_repeats": 3, "num_repeats_completed": 3, "is_complete": True,
        "num_questions": 10, "accuracy_mean": accuracy_mean, "accuracy_std": 0.01,
        "output_tokens_per_sec_mean": 100.0, "input_tokens_per_sec_mean": 200.0,
        "queries_per_sec_mean": 1.0, "mean_reasoning_tokens_mean": 500.0,
    }


class TestBuildFinalTable(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_joins_train_and_eval_summaries(self):
        train_dir = os.path.join(self.tmpdir.name, "results")
        eval_dir = os.path.join(self.tmpdir.name, "eval_results")
        _write_json(
            os.path.join(train_dir, "s1k11_all_qwen3_cotgen_summary.json"),
            {
                "dataset_name": "s1k11_all_qwen3", "mode": "cotgen",
                "train_results": {"total_flos": 99, "train_runtime": 10.0, "num_input_tokens_seen": 1000},
            },
        )
        _write_json(
            os.path.join(eval_dir, "s1k11_all_qwen3_cotgen_aime24_summary.json"),
            _eval_summary("s1k11_all_qwen3", "cotgen", "aime24"),
        )
        out_csv = os.path.join(self.tmpdir.name, "final.csv")
        n = tables.build_final_table(train_dir, [eval_dir], out_csv)
        self.assertEqual(n, 1)
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["model"], "qwen3")
        self.assertEqual(rows[0]["ablation"], "all")
        self.assertEqual(rows[0]["train_total_flos"], "99")
        self.assertEqual(rows[0]["train_tokens_per_sec"], "100.0")

    def test_base_row_without_train_summary_no_warning_crash(self):
        eval_dir = os.path.join(self.tmpdir.name, "eval_results")
        _write_json(
            os.path.join(eval_dir, "base_qwen3_base_aime24_summary.json"),
            _eval_summary("base_qwen3", "base", "aime24"),
        )
        out_csv = os.path.join(self.tmpdir.name, "final.csv")
        n = tables.build_final_table(os.path.join(self.tmpdir.name, "results"), [eval_dir], out_csv)
        self.assertEqual(n, 1)

    def test_no_eval_summaries_returns_zero(self):
        n = tables.build_final_table(
            os.path.join(self.tmpdir.name, "results"),
            [os.path.join(self.tmpdir.name, "eval_results")],
            os.path.join(self.tmpdir.name, "final.csv"),
        )
        self.assertEqual(n, 0)


class TestBuildPivotTable(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_pivot_has_one_row_per_model_ablation_benchmark(self):
        eval_dir = os.path.join(self.tmpdir.name, "eval_results")
        _write_json(
            os.path.join(eval_dir, "s1k11_all_qwen3_cotgen_aime24_summary.json"),
            _eval_summary("s1k11_all_qwen3", "cotgen", "aime24", accuracy_mean=0.4),
        )
        _write_json(
            os.path.join(eval_dir, "s1k11_all_qwen3_cotcond_aime24_summary.json"),
            _eval_summary("s1k11_all_qwen3", "cotcond", "aime24", accuracy_mean=0.6),
        )
        pivot = tables.build_pivot_table([eval_dir], models=["qwen3"])
        self.assertEqual(len(pivot), len(tables.PIVOT_ABLATIONS) * len(tables.PIVOT_BENCHMARKS))
        row = pivot[(pivot["ablation"] == "all") & (pivot["benchmark"] == "aime24")].iloc[0]
        self.assertAlmostEqual(row["cotgen_acc"], 0.4)
        self.assertAlmostEqual(row["cotcond_acc"], 0.6)
        # (0.6 - 0.4) / 0.4 * 100 = 50.0
        self.assertAlmostEqual(row["delta_cotcond_vs_cotgen_pct"], 50.0)

    def test_missing_cell_is_none_not_crash(self):
        pivot = tables.build_pivot_table([os.path.join(self.tmpdir.name, "eval_results")], models=["qwen3"])
        self.assertEqual(len(pivot), len(tables.PIVOT_ABLATIONS) * len(tables.PIVOT_BENCHMARKS))
        self.assertTrue(pivot["cotgen_acc"].isna().all())


if __name__ == "__main__":
    unittest.main()
