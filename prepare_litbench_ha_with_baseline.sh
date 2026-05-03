#!/bin/bash

# Convert the exact LitBench-with-baseline Hugging Face dataset into the parquet
# directories expected by the Qwen dense SDPO launcher.
#
# Train and val reward types should normally MATCH. The original setup used
# train=margin / val=absolute, which produced specification gaming (the policy
# learned to inflate rubric strictness for bigger margin reward, while the
# absolute-shaped val reward stayed flat). Defaulting both sides to
# criteria_margin gives finer per-criterion gradient signal, forces structured
# <criterion> rubric output, and keeps train/val on the same shape.
#
# Override via env vars:
#   JUDGE_REWARD_TYPE_TRAIN=criteria_margin        ./prepare_litbench_ha_with_baseline.sh
#   JUDGE_REWARD_TYPE_TRAIN=criteria_margin_bounded ./prepare_litbench_ha_with_baseline.sh
#   JUDGE_REWARD_TYPE_TRAIN=margin JUDGE_REWARD_TYPE_VAL=absolute ./prepare_litbench_ha_with_baseline.sh  # legacy

set -eo pipefail

DATASET_NAME="${DATASET_NAME:-sumuks/litbench-ha-with-baseline}"
JUDGE_REWARD_TYPE_TRAIN="${JUDGE_REWARD_TYPE_TRAIN:-criteria_margin}"
JUDGE_REWARD_TYPE_VAL="${JUDGE_REWARD_TYPE_VAL:-criteria_margin}"

# Derive parquet folder names from the reward type so the path is unambiguous.
# Override TRAIN_DATA/EVAL_DATA explicitly to keep the old folder names if needed.
TRAIN_DATA="${TRAIN_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_${JUDGE_REWARD_TYPE_TRAIN}}"
EVAL_DATA="${EVAL_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_${JUDGE_REWARD_TYPE_VAL}}"

echo "----------------------------------------------------------------"
echo "Preprocessing LitBench dataset for SDPO"
echo "Dataset:                  $DATASET_NAME"
echo "Train reward type:        $JUDGE_REWARD_TYPE_TRAIN"
echo "Val reward type:          $JUDGE_REWARD_TYPE_VAL"
echo "Train output:             $TRAIN_DATA"
echo "Eval output:              $EVAL_DATA"
echo "----------------------------------------------------------------"

python -m environments.dpo_to_rupo_verl.preprocess_dataset \
  --dataset_name "$DATASET_NAME" \
  --policy_prompt_mode pair \
  --judge_reward_type "$JUDGE_REWARD_TYPE_TRAIN" \
  --output_dir "$TRAIN_DATA"

# Skip the second preprocess if train and val use the same reward type AND the
# same output dir. Otherwise produce a separate val parquet.
if [[ "$JUDGE_REWARD_TYPE_TRAIN" != "$JUDGE_REWARD_TYPE_VAL" || "$TRAIN_DATA" != "$EVAL_DATA" ]]; then
  python -m environments.dpo_to_rupo_verl.preprocess_dataset \
    --dataset_name "$DATASET_NAME" \
    --policy_prompt_mode pair \
    --judge_reward_type "$JUDGE_REWARD_TYPE_VAL" \
    --output_dir "$EVAL_DATA"
fi
