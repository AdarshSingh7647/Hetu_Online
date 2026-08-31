"""
hetu-online -- CLI entry point.

  hetu-online build-data        ...   # raw {prompt,thinking,answer} -> ShareGPT data for both modes
  hetu-online make-config       ...   # generate a LLaMA-Factory YAML for one mode
  hetu-online train             ...   # launch llamafactory-cli with the right env var for the mode
  hetu-online run-one           ...   # train + log + summarize one (dataset, mode) job in one call
  hetu-online summarize-training ...  # join every training summary JSON into one CSV
  hetu-online eval              ...   # run one (checkpoint, benchmark) eval job via vLLM
  hetu-online build-eval-table  ...   # join training + eval summaries into one wide CSV
  hetu-online build-pivot-table ...   # print a base/cotgen/cotcond accuracy comparison matrix
"""

from __future__ import annotations

import argparse
import sys

from .config_builder import build_config
from .data_builder import build_cot_data
from .eval.run_eval import add_eval_arguments, run_eval_from_args
from .eval.tables import register_table_subcommands
from .model_families import MODEL_FAMILIES
from .orchestration.run_one import _add_run_one_parser
from .orchestration.summarize import _add_summarize_parser
from .train import run_training

# eval.benchmarks/eval.run_eval/eval.tables only import datasets/
# transformers/vllm/pandas LAZILY, inside the functions that actually need
# them (not at module level) -- so importing them here to build the
# `eval`/`build-eval-table`/`build-pivot-table` subparsers' --help text and
# argparse choices works even in a trainer-only environment that never
# installed vLLM.

# CLI-level gate, narrower than MODEL_FAMILIES: gemma4's FamilyConfig,
# vendored liger patch, and tests all exist in this repo, but gemma4
# training hasn't completed a validated end-to-end run on real GPU
# hardware yet (see model_families.py for the template/tag facts already
# verified from source -- what's NOT yet verified is that a full training
# run actually succeeds with them). Blocking it here, one line, rather
# than in model_families.py itself, means re-enabling it after GPU
# validation is exactly that -- flipping this tuple back to
# MODEL_FAMILIES -- with no other code changes required.
CLI_ENABLED_MODEL_FAMILIES = ("qwen3",)


def _add_build_data_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-data",
        help="Build cotgen + cotcond ShareGPT datasets from raw prompt/thinking/answer examples.",
        description=(
            "Turns raw {prompt, thinking, answer} examples into LLaMA-Factory ShareGPT SFT "
            "data for both CotGen and CotCond, and registers them in dataset_info.json."
        ),
    )
    p.add_argument("--input", required=True, help="Path to raw train examples (JSON array or .jsonl).")
    p.add_argument("--val_input", default=None, help="Path to raw val examples. If omitted, a slice of --input is held out.")
    p.add_argument("--val_fraction", type=float, default=0.02, help="Used only when --val_input is not given.")
    p.add_argument("--out_dir", required=True, help="Where to write the *_cotgen_*/*_cotcond_* JSON files.")
    p.add_argument("--dataset_name", required=True, help="Short task tag, e.g. 'my_task'.")
    p.add_argument("--llamafactory_data_dir", required=True, help="Path to LLaMA-Factory/data.")
    p.add_argument(
        "--model_family", required=True, choices=CLI_ENABLED_MODEL_FAMILIES,
        help="Selects the reasoning-span tags written into the data ('<think>' for qwen3, "
        "'<|channel>thought' for gemma4). MUST match --model_family passed to "
        "`hetu-online train`, or CotCond's masking silently fails open to full supervision.",
    )
    p.add_argument("--system_prompt", default="You are a careful, helpful assistant.")
    p.add_argument("--seed", type=int, default=12345)
    p.set_defaults(func=_run_build_data)


def _run_build_data(args: argparse.Namespace) -> int:
    written = build_cot_data(
        input_path=args.input,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        llamafactory_data_dir=args.llamafactory_data_dir,
        model_family=args.model_family,
        val_input=args.val_input,
        val_fraction=args.val_fraction,
        system_prompt=args.system_prompt,
        seed=args.seed,
    )
    print("\n[hetu-online] Done. Registered dataset keys:")
    for key in written:
        print(f"  - {key}")
    print(f"\nNext: hetu-online make-config --dataset_name {args.dataset_name} "
          f"--model_family {args.model_family} --model_path <path> --output_dir <dir> "
          f"--mode cotgen   (or cotcond)")
    return 0


def _add_make_config_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "make-config",
        help="Generate a LLaMA-Factory LoRA SFT YAML for one mode (cotgen or cotcond).",
    )
    p.add_argument("--dataset_name", required=True, help="Must match --dataset_name used in build-data.")
    p.add_argument("--model_path", required=True, help="Local path or HF repo id of the base model.")
    p.add_argument("--output_dir", required=True, help="Checkpoint output dir for this run.")
    p.add_argument("--mode", required=True, choices=["cotgen", "cotcond"])
    p.add_argument(
        "--model_family", required=True, choices=CLI_ENABLED_MODEL_FAMILIES,
        help="Selects the validated template + freeze/flash_attn/liger/cutoff_len settings "
        "for this family (see model_families.py). MUST match --model_family used in "
        "build-data for this --dataset_name.",
    )
    p.add_argument("--out_config", default=None, help="Where to write the YAML. Default: ./configs/<dataset_name>_<mode>.yaml")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument(
        "--cutoff_len", type=int, default=None,
        help="Overrides the model_family's validated default cutoff_len. Leave unset unless "
        "you have a specific reason -- the default was tuned per family (e.g. Gemma-4's "
        "28672 vs Qwen3's 32768) against real tokenized data length and GPU memory.",
    )
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", default="0.0001")
    p.add_argument("--num_train_epochs", default="2.0")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    p.set_defaults(func=_run_make_config)


def _run_make_config(args: argparse.Namespace) -> int:
    build_config(
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        mode=args.mode,
        model_family=args.model_family,
        out_config=args.out_config,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        cutoff_len=args.cutoff_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        seed=args.seed,
    )
    return 0


def _add_train_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "train",
        help="Launch llamafactory-cli with the right PYTHONPATH/env var for the given mode.",
        description=(
            "IMPORTANT: --model_family (and any other optional flag this command grows in "
            "the future) MUST be given BEFORE the positional args (dataset_name mode "
            "config_path), e.g. "
            "`hetu-online train --model_family gemma4 my_dataset cotcond cfg.yaml`. "
            "extra_args uses argparse.REMAINDER, which greedily captures everything after the "
            "first positional -- a flag placed after config_path silently lands in extra_args "
            "(passed through to llamafactory-cli, which ignores it) instead of erroring."
        ),
    )
    p.add_argument("dataset_name", help="Task/dataset tag (used only for the log line).")
    p.add_argument("mode", choices=["cotgen", "cotcond"])
    p.add_argument("config_path", help="Path to the YAML config (from make-config).")
    p.add_argument(
        "--model_family", required=True, choices=CLI_ENABLED_MODEL_FAMILIES,
        help="Selects HETU_THINK_OPEN_TAG/HETU_THINK_CLOSE_TAG for CotCond masking. MUST "
        "match --model_family used in build-data for this dataset, or masking silently "
        "fails open to full supervision. MUST be given before the positional args -- see "
        "this command's description.",
    )
    p.add_argument("extra_args", nargs=argparse.REMAINDER,
                    help="Extra args passed through to llamafactory-cli train.")
    p.set_defaults(func=_run_train)


def _run_train(args: argparse.Namespace) -> int:
    return run_training(
        dataset_name=args.dataset_name,
        mode=args.mode,
        config_path=args.config_path,
        model_family=args.model_family,
        extra_args=args.extra_args,
    )


def _add_eval_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "eval",
        help="Run one (checkpoint, benchmark) eval job via vLLM.",
    )
    add_eval_arguments(p)
    p.set_defaults(func=run_eval_from_args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hetu-online", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_build_data_parser(subparsers)
    _add_make_config_parser(subparsers)
    _add_train_parser(subparsers)
    _add_run_one_parser(subparsers)
    _add_summarize_parser(subparsers)
    _add_eval_parser(subparsers)
    register_table_subcommands(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
