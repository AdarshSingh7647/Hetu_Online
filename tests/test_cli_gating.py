"""Tests that the CLI-level model_family gate actually blocks gemma4 while
model_families.py itself still has it -- this is the exact mechanism this
repo relies on to ship gemma4's (untested) code without exposing it as a
runnable CLI option yet."""

import unittest

from hetu_online import model_families
from hetu_online.cli import CLI_ENABLED_MODEL_FAMILIES, main


class TestCliGating(unittest.TestCase):
    def test_gate_is_narrower_than_full_registry(self):
        self.assertEqual(CLI_ENABLED_MODEL_FAMILIES, ("qwen3",))
        self.assertIn("gemma4", model_families.MODEL_FAMILIES)
        self.assertNotIn("gemma4", CLI_ENABLED_MODEL_FAMILIES)

    def test_gemma4_rejected_by_build_data_cli(self):
        with self.assertRaises(SystemExit):
            main([
                "build-data", "--input", "x.json", "--out_dir", "y",
                "--dataset_name", "z", "--llamafactory_data_dir", "d",
                "--model_family", "gemma4",
            ])

    def test_gemma4_rejected_by_make_config_cli(self):
        with self.assertRaises(SystemExit):
            main([
                "make-config", "--dataset_name", "z", "--model_path", "m",
                "--output_dir", "o", "--mode", "cotgen", "--model_family", "gemma4",
            ])

    def test_gemma4_rejected_by_train_cli(self):
        with self.assertRaises(SystemExit):
            main(["train", "--model_family", "gemma4", "z", "cotgen", "cfg.yaml"])

    def test_qwen3_accepted_by_build_data_parser(self):
        # Parses fine (fails later, deeper, on the missing input file --
        # not an argparse-level rejection), confirming qwen3 itself isn't
        # blocked by the gate.
        import argparse

        from hetu_online.cli import _add_build_data_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        _add_build_data_parser(subparsers)
        args = parser.parse_args([
            "build-data", "--input", "x.json", "--out_dir", "y",
            "--dataset_name", "z", "--llamafactory_data_dir", "d",
            "--model_family", "qwen3",
        ])
        self.assertEqual(args.model_family, "qwen3")


if __name__ == "__main__":
    unittest.main()
