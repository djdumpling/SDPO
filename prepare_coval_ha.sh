#!/bin/bash

# Convert the CoVal HA preference dataset into the parquet directories expected
# by the HA SDPO launcher.
#
# Assumes the SDPO virtualenv is already active. Use python directly so uv does
# not re-resolve optional project extras during data preparation.

set -eo pipefail

DATASET_NAME="${DATASET_NAME:-sumuks/coval-ha}"
TRAIN_DATA="${TRAIN_DATA:-datasets/ha_coval_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/ha_coval_absolute}"
ENABLE_THINKING_PROMPT="${ENABLE_THINKING_PROMPT:-true}"

echo "----------------------------------------------------------------"
echo "Preprocessing HA preference dataset for SDPO"
echo "Dataset: $DATASET_NAME"
echo "Train output: $TRAIN_DATA"
echo "Eval output: $EVAL_DATA"
echo "Thinking-aware prompt: $ENABLE_THINKING_PROMPT"
echo "----------------------------------------------------------------"

python -m environments.ha_rubric_verl.preprocess_dataset \
  --dataset_name "$DATASET_NAME" \
  --policy_prompt_mode pair \
  --judge_reward_type margin \
  --enable_thinking_prompt "$ENABLE_THINKING_PROMPT" \
  --output_dir "$TRAIN_DATA"

python -m environments.ha_rubric_verl.preprocess_dataset \
  --dataset_name "$DATASET_NAME" \
  --policy_prompt_mode pair \
  --judge_reward_type absolute \
  --enable_thinking_prompt "$ENABLE_THINKING_PROMPT" \
  --output_dir "$EVAL_DATA"
