#!/bin/bash

# Train SDPO rubric generation with Qwen dense models on CoVal HA preference pairs.
#
# Usage:
#   # 1. Preprocess CoVal HA into verl parquet.
#   ./prepare_coval_ha.sh
#
#   # 2. Run through a wrapper that starts a local vLLM judge, or export:
#   #   OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY

CONFIG_NAME="sdpo_ha"
DATASET_NAME="${DATASET_NAME:-sumuks/coval-ha}"
TRAIN_DATA="${TRAIN_DATA:-datasets/ha_coval_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/ha_coval_absolute}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-6}
ROLLOUT_N=${ROLLOUT_N:-4}
LR=${LR:-5e-6}
ALPHA=${ALPHA:-0.5}

ENABLE_THINKING="${ENABLE_THINKING:-true}"
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_REPROMPT_LEN=${MAX_REPROMPT_LEN:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-16384}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-24576}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

TP_SIZE=${TP_SIZE:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-6}
export NNODES=${NNODES:-1}

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${JUDGE_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_PATH}"
export JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-false}"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-32}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-900}"
export RUBRIC_FORMAT_REWARD_MAX="${RUBRIC_FORMAT_REWARD_MAX:-0.1}"

SUFFIX=${1:-"qwen35_9b_coval_ha_local_judge"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

if [[ -z "$OPENAI_BASE_URL" ]]; then
    echo "Missing judge endpoint."
    echo "Run through a wrapper that starts vLLM, or export OPENAI_BASE_URL before invoking this helper directly."
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
data.apply_chat_template_kwargs={enable_thinking:$ENABLE_THINKING} \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
trainer.group_name=SDPO-ha \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$NNODES \
trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.lora_rank=$LORA_RANK \
actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
actor_rollout_ref.model.target_modules=all-linear \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_REPROMPT_LEN \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=8 \
+ray_kwargs.ray_init.include_dashboard=False"

echo "----------------------------------------------------------------"
echo "Starting SDPO HA Rubric Training (Qwen dense + LoRA)"
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET_NAME"
echo "Train data: $TRAIN_DATA (margin reward)"
echo "Eval data:  $EVAL_DATA (absolute reward)"
echo "Model: $MODEL_PATH"
echo "Judge endpoint: $OPENAI_BASE_URL"
echo "Judge model: $JUDGE_MODEL  judge thinking: $JUDGE_ENABLE_THINKING"
echo "Format reward max: $RUBRIC_FORMAT_REWARD_MAX  rollout dump: $ROLLOUT_DATA_DIR"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$TRAIN_DATA" $ARGS
