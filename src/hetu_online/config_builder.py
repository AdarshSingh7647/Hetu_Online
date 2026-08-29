"""
HETU_Online config generator -- writes a LLaMA-Factory SFT YAML for either
'cotgen' or 'cotcond' mode, from one shared template. The two modes differ
in exactly THREE fields (dataset key, eval_dataset key, output_dir) --
everything else (LoRA rank, batch size, lr, cutoff_len...) is identical.

CotCond additionally REQUIRES an env var at launch time -- this is NOT a
YAML key, LLaMA-Factory has no concept of it. `hetu-online train` sets it
for you automatically based on mode.
"""

from __future__ import annotations

import os
from typing import Optional

TEMPLATE = """\
### model
model_name_or_path: {model_path}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {lora_rank}
lora_alpha: {lora_alpha}
lora_target: all

### dataset
dataset: {dataset_name}_{mode}_train
template: {template}
enable_thinking: true
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### mask_history
mask_history: false

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: {save_steps}
save_total_limit: 2
plot_loss: true
overwrite_output_dir: false
report_to: tensorboard

### train
per_device_train_batch_size: {per_device_train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: {learning_rate}
num_train_epochs: {num_train_epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
seed: {seed}
ddp_timeout: 180000000

### eval
val_size: 0.0
per_device_eval_batch_size: {per_device_train_batch_size}
eval_strategy: steps
eval_dataset: {dataset_name}_{mode}_val
eval_steps: {save_steps}
{requires_comment}
"""

COTCOND_REQUIRES_COMMENT = (
    "\n# REQUIRES: HETU_THINK_CONTENT_MASK=\"1\" exported at launch (not a YAML key --\n"
    "# see hetu_online/sitecustomize.py). Masks loss on the <think>...</think> reasoning\n"
    "# CONTENT only; the tags and the final answer stay supervised. Use `hetu-online train`\n"
    "# instead of raw llamafactory-cli so this is set for you automatically.\n"
)

COTGEN_COMMENT = (
    "\n# cotgen: full loss on the whole response (reasoning + answer), no env var needed.\n"
)


def build_config(
    dataset_name: str,
    model_path: str,
    output_dir: str,
    mode: str,
    out_config: Optional[str] = None,
    template: str = "qwen3",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    cutoff_len: int = 4096,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 16,
    learning_rate: str = "0.0001",
    num_train_epochs: str = "2.0",
    save_steps: int = 200,
    seed: int = 12345,
    configs_dir: Optional[str] = None,
) -> str:
    if mode not in ("cotgen", "cotcond"):
        raise ValueError(f"mode must be 'cotgen' or 'cotcond', got {mode!r}")

    if out_config is None:
        base_dir = configs_dir or os.path.join(os.getcwd(), "configs")
        out_config = os.path.join(base_dir, f"{dataset_name}_{mode}.yaml")

    requires_comment = COTCOND_REQUIRES_COMMENT if mode == "cotcond" else COTGEN_COMMENT

    content = TEMPLATE.format(
        model_path=model_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        dataset_name=dataset_name,
        mode=mode,
        template=template,
        cutoff_len=cutoff_len,
        output_dir=output_dir,
        save_steps=save_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        seed=seed,
        requires_comment=requires_comment,
    )

    os.makedirs(os.path.dirname(out_config), exist_ok=True)
    with open(out_config, "w") as f:
        f.write(content)

    print(f"[hetu-online] wrote {out_config}")
    if mode == "cotcond":
        print("[hetu-online] Reminder: this mode needs HETU_THINK_CONTENT_MASK=1 at launch. "
              "Use `hetu-online train` instead of raw llamafactory-cli, or export it yourself.")

    return out_config
