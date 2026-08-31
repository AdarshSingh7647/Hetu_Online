"""Tests for config_builder.py, scoped to the two supported model families."""

import os
import tempfile
import unittest

import yaml

from hetu_online import config_builder


class TestBuildConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _build(self, model_family, mode="cotcond", **overrides):
        out_config = os.path.join(self.tmpdir.name, f"{model_family}_{mode}.yaml")
        kwargs = dict(
            dataset_name="task",
            model_path="some/model",
            output_dir="/ckpt/task",
            mode=mode,
            model_family=model_family,
            out_config=out_config,
        )
        kwargs.update(overrides)
        config_builder.build_config(**kwargs)
        with open(out_config) as f:
            return yaml.safe_load(f)

    def test_qwen3_has_no_gemma4_specific_keys(self):
        cfg = self._build("qwen3")
        self.assertEqual(cfg["template"], "qwen3")
        self.assertEqual(cfg["cutoff_len"], 32768)
        self.assertNotIn("freeze_vision_tower", cfg)
        self.assertNotIn("freeze_multi_modal_projector", cfg)
        self.assertNotIn("flash_attn", cfg)
        self.assertNotIn("enable_liger_kernel", cfg)

    def test_gemma4_has_all_required_keys(self):
        cfg = self._build("gemma4")
        self.assertEqual(cfg["template"], "gemma4")
        self.assertNotEqual(cfg["template"], "gemma4n")
        self.assertEqual(cfg["cutoff_len"], 28672)
        self.assertTrue(cfg["freeze_vision_tower"])
        self.assertTrue(cfg["freeze_multi_modal_projector"])
        self.assertEqual(cfg["flash_attn"], "fa2")
        self.assertTrue(cfg["enable_liger_kernel"])

    def test_cutoff_len_override_respected(self):
        cfg = self._build("gemma4", cutoff_len=16384)
        self.assertEqual(cfg["cutoff_len"], 16384)

    def test_dataset_keys_use_mode_suffix(self):
        cfg = self._build("qwen3", mode="cotgen")
        self.assertEqual(cfg["dataset"], "task_cotgen_train")
        self.assertEqual(cfg["eval_dataset"], "task_cotgen_val")

    def test_unsupported_family_raises(self):
        with self.assertRaises(ValueError):
            self._build("llama3")


if __name__ == "__main__":
    unittest.main()
