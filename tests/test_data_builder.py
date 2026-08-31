"""Tests for data_builder.py, scoped to the two supported model families."""

import json
import os
import tempfile
import unittest

from hetu_online import data_builder


RAW_EXAMPLES = [
    {"prompt": "What is 2+2?", "thinking": "2 plus 2 equals 4.", "answer": "4"},
    {"prompt": "What is 3+3?", "thinking": "3 plus 3 equals 6.", "answer": "6"},
    {"prompt": "What is 5+5?", "thinking": "5 plus 5 equals 10.", "answer": "10"},
    {"prompt": "What is 7+7?", "thinking": "7 plus 7 equals 14.", "answer": "14"},
]


class TestBuildThinkResponse(unittest.TestCase):
    def test_qwen3_tags(self):
        think_open, think_close = data_builder.DEFAULT_THINK_OPEN, data_builder.DEFAULT_THINK_CLOSE
        resp = data_builder.build_think_response("reasoning here", "42", think_open, think_close)
        self.assertEqual(resp, "<think>\nreasoning here\n</think>\n\n42")

    def test_gemma4_tags_no_leading_newline_on_close(self):
        from hetu_online import model_families

        think_open, think_close = model_families.think_tags("gemma4")
        resp = data_builder.build_think_response("reasoning here", "42", think_open, think_close)
        self.assertEqual(resp, "<|channel>thought\nreasoning here<channel|>42")
        # Regression guard: a "symmetrized" close tag would insert a
        # newline before <channel|>, which would not match what
        # LLaMA-Factory's gemma4 template actually renders.
        self.assertNotIn("\n<channel|>", resp)


class TestBuildCotData(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.input_path = os.path.join(self.tmpdir.name, "raw.json")
        with open(self.input_path, "w") as f:
            json.dump(RAW_EXAMPLES, f)
        self.out_dir = os.path.join(self.tmpdir.name, "data", "task")
        self.lf_data_dir = os.path.join(self.tmpdir.name, "data")

    def test_qwen3_and_gemma4_produce_byte_identical_shape_different_tags(self):
        for family in ("qwen3", "gemma4"):
            written = data_builder.build_cot_data(
                input_path=self.input_path,
                out_dir=os.path.join(self.out_dir, family),
                dataset_name=f"task_{family}",
                llamafactory_data_dir=self.lf_data_dir,
                model_family=family,
                val_fraction=0.25,
                seed=1,
            )
            self.assertEqual(
                set(written.keys()),
                {
                    f"task_{family}_cotgen_train", f"task_{family}_cotgen_val",
                    f"task_{family}_cotcond_train", f"task_{family}_cotcond_val",
                },
            )
            with open(written[f"task_{family}_cotgen_train"]) as f:
                cotgen_train = json.load(f)
            with open(written[f"task_{family}_cotcond_train"]) as f:
                cotcond_train = json.load(f)
            # cotgen/cotcond are byte-identical data -- only the training-time
            # env var differs. Verifying that invariant survives per-family
            # tag substitution.
            self.assertEqual(cotgen_train, cotcond_train)

        with open(written[f"task_gemma4_cotgen_train"]) as f:
            gemma4_train = json.load(f)
        qwen3_path = os.path.join(self.out_dir, "qwen3", "task_qwen3_cotgen_train.json")
        with open(qwen3_path) as f:
            qwen3_train = json.load(f)
        self.assertIn("<think>", qwen3_train[0]["conversations"][1]["value"])
        self.assertIn("<|channel>thought", gemma4_train[0]["conversations"][1]["value"])

    def test_registers_datasets_in_dataset_info(self):
        data_builder.build_cot_data(
            input_path=self.input_path,
            out_dir=self.out_dir,
            dataset_name="task_qwen3",
            llamafactory_data_dir=self.lf_data_dir,
            model_family="qwen3",
            val_fraction=0.25,
            seed=1,
        )
        info_path = os.path.join(self.lf_data_dir, "dataset_info.json")
        with open(info_path) as f:
            info = json.load(f)
        self.assertIn("task_qwen3_cotgen_train", info)
        self.assertEqual(info["task_qwen3_cotgen_train"]["formatting"], "sharegpt")

    def test_unsupported_family_raises(self):
        with self.assertRaises(ValueError):
            data_builder.build_cot_data(
                input_path=self.input_path,
                out_dir=self.out_dir,
                dataset_name="task_x",
                llamafactory_data_dir=self.lf_data_dir,
                model_family="llama3",
            )


if __name__ == "__main__":
    unittest.main()
