#!/bin/bash

# Convert the exact LitBench-with-baseline Hugging Face dataset into the parquet
# directories expected by the Qwen3.5-9B SDPO launcher.

set -eo pipefail

DATASET_NAME="${DATASET_NAME:-sumuks/litbench-ha-with-baseline}"
TRAIN_DATA="${TRAIN_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_absolute}"

echo "----------------------------------------------------------------"
echo "Preprocessing LitBench dataset for SDPO"
echo "Dataset: $DATASET_NAME"
echo "Train output: $TRAIN_DATA"
echo "Eval output: $EVAL_DATA"
echo "----------------------------------------------------------------"

uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \
  --dataset_name "$DATASET_NAME" \
  --policy_prompt_mode pair \
  --judge_reward_type margin \
  --output_dir "$TRAIN_DATA"

uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \
  --dataset_name "$DATASET_NAME" \
  --policy_prompt_mode pair \
  --judge_reward_type absolute \
  --output_dir "$EVAL_DATA"
