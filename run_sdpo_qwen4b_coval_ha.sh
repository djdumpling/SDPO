#!/bin/bash

# Train SDPO rubric generation with Qwen3-4B-Instruct on CoVal HA preference pairs.
#
# Usage:
#   # 1. Preprocess CoVal HA into verl parquet.
#   ./prepare_coval_ha.sh
#
#   # 2. Export an OpenAI-compatible judge endpoint, then run:
#   #   OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY ./run_sdpo_qwen4b_coval_ha.sh

CONFIG_NAME="sdpo_ha"
DATASET_NAME="${DATASET_NAME:-sumuks/coval-ha}"
TRAIN_DATA="${TRAIN_DATA:-datasets/ha_coval_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/ha_coval_absolute}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
LR=${LR:-5e-6}
ALPHA=${ALPHA:-0.5}

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

TP_SIZE=${TP_SIZE:-4}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${JUDGE_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
export JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-false}"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-64}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-900}"
export RUBRIC_FORMAT_REWARD_MAX="${RUBRIC_FORMAT_REWARD_MAX:-0.1}"

SUFFIX=${1:-"qwen4b_coval_ha"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

if [[ -z "$OPENAI_BASE_URL" && "$JUDGE_MODEL" != gpt-* ]]; then
    echo "Missing judge endpoint."
    echo "Export OPENAI_BASE_URL for a local/OpenAI-compatible judge, or set JUDGE_MODEL to an OpenAI-hosted model and provide OPENAI_API_KEY."
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/$TRAIN_DATA/train.parquet" || ! -f "$PROJECT_ROOT/$EVAL_DATA/test.parquet" ]]; then
    echo "Missing preprocessed CoVal HA parquet files."
    echo "Run this from $PROJECT_ROOT:"
    echo "  ./prepare_coval_ha.sh"
    exit 1
fi

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="SDPO-HA-bs${TRAIN_BATCH_SIZE}-n${ROLLOUT_N}-alpha${ALPHA}-lr${LR}-lora${LORA_RANK}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.train_files=[$TRAIN_DATA/train.parquet] \
data.val_files=[$EVAL_DATA/test.parquet] \
data.apply_chat_template_kwargs={enable_thinking:false} \
trainer.group_name=SDPO-ha \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$NNODES \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.lora_rank=$LORA_RANK \
actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
actor_rollout_ref.model.target_modules=all-linear \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=8 \
+ray_kwargs.ray_init.include_dashboard=False"

echo "----------------------------------------------------------------"
echo "Starting SDPO HA Rubric Training (Qwen3-4B-Instruct + LoRA)"
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET_NAME"
echo "Train data: $TRAIN_DATA (margin reward)"
echo "Eval data:  $EVAL_DATA (absolute reward)"
echo "Model: $MODEL_PATH"
echo "Judge: $JUDGE_MODEL"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$TRAIN_DATA" $ARGS
