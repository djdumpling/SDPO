#!/bin/bash

# Train rubric generation with SDPO self-distillation on DPO preference data.
#
# Usage:
#   # 1. Preprocess the dataset (only needed once):
#   python -m environments.dpo_to_rupo_verl.preprocess_dataset \
#       --dataset_name sumuks/litbench-ha \
#       --policy_prompt_mode pair \
#       --judge_reward_type margin \
#       --output_dir datasets/dpo_to_rupo
#
#   # 2. Run training:
#   ./run_sdpo_rupo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo_rupo"
DATA_PATH="datasets/dpo_to_rupo"

# Model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# Training hyperparameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
LR=${LR:-5e-6}
ALPHA=${ALPHA:-0.5}

# Judge configuration (via environment variables read by reward_fn.py)
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-64}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-900}"
# OPENAI_API_KEY and OPENAI_BASE_URL should be set in the environment

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}

SUFFIX=${1:-"sdpo_rupo"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="SDPO-RUPO-bs${TRAIN_BATCH_SIZE}-n${ROLLOUT_N}-alpha${ALPHA}-lr${LR}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-rupo \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=8"

echo "----------------------------------------------------------------"
echo "Starting SDPO Rubric Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Judge: $JUDGE_MODEL"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
