"""Regression tests for model_families.py -- these lock in facts that fail
SILENTLY if wrong (see module docstring), so a test catching an accidental
change is the whole point of this file."""

import unittest

from hetu_online import model_families


class TestModelFamilies(unittest.TestCase):
    def test_supported_families(self):
        self.assertEqual(model_families.MODEL_FAMILIES, ("qwen3", "gemma4"))

    def test_unsupported_family_raises(self):
        with self.assertRaises(ValueError):
            model_families.get("gpt2")
        with self.assertRaises(ValueError):
            model_families.get("gemma3")
        with self.assertRaises(ValueError):
            model_families.get("gemma4n")

    def test_qwen3_config(self):
        cfg = model_families.get("qwen3")
        self.assertEqual(cfg.template, "qwen3")
        self.assertEqual(cfg.think_open, "<think>\n")
        self.assertEqual(cfg.think_close, "\n</think>\n\n")
        self.assertFalse(cfg.freeze_vision)
        self.assertIsNone(cfg.flash_attn)
        self.assertFalse(cfg.enable_liger_kernel)
        self.assertEqual(cfg.default_cutoff_len, 32768)

    def test_gemma4_config(self):
        cfg = model_families.get("gemma4")
        # NOT gemma4n -- see module docstring for why gemma4n crashes on
        # text-only training (fake-audio injection under flash-attn).
        self.assertEqual(cfg.template, "gemma4")
        self.assertNotEqual(cfg.template, "gemma4n")
        self.assertEqual(cfg.think_open, "<|channel>thought\n")
        # No leading newline on the close tag -- do not "symmetrize" this.
        self.assertEqual(cfg.think_close, "<channel|>")
        self.assertTrue(cfg.freeze_vision)
        self.assertEqual(cfg.flash_attn, "fa2")
        self.assertTrue(cfg.enable_liger_kernel)
        self.assertEqual(cfg.default_cutoff_len, 28672)

    def test_think_tags_helper_matches_get(self):
        for family in model_families.MODEL_FAMILIES:
            cfg = model_families.get(family)
            self.assertEqual(model_families.think_tags(family), (cfg.think_open, cfg.think_close))

    def test_template_helper_matches_get(self):
        for family in model_families.MODEL_FAMILIES:
            self.assertEqual(model_families.template(family), model_families.get(family).template)


if __name__ == "__main__":
    unittest.main()
