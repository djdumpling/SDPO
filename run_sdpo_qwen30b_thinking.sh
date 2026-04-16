#!/bin/bash

# Train SDPO rubric generation with Qwen3-30B-A3B-Thinking on 4x H200 using LoRA.
#
# Usage:
#   # 1. Preprocess LitBench twice (once per reward type):
#   python environments/dpo_to_rupo_verl/preprocess_dataset.py \
#       --dataset_name sumuks/litbench-ha --policy_prompt_mode pair \
#       --judge_reward_type margin --output_dir datasets/dpo_to_rupo_margin
#   python environments/dpo_to_rupo_verl/preprocess_dataset.py \
#       --dataset_name sumuks/litbench-ha --policy_prompt_mode pair \
#       --judge_reward_type absolute --output_dir datasets/dpo_to_rupo_absolute
#
#   # 2. Run training (from this dir):
#   ./run_sdpo_qwen30b_thinking.sh [experiment_name_suffix]

CONFIG_NAME="sdpo_rupo"
TRAIN_DATA="datasets/dpo_to_rupo_margin"
EVAL_DATA="datasets/dpo_to_rupo_absolute"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Thinking-2507}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
LR=${LR:-5e-6}
ALPHA=${ALPHA:-0.5}

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

TP_SIZE=${TP_SIZE:-4}

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}

export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-64}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-900}"

SUFFIX=${1:-"qwen30b_a3b_thinking_litbench"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="SDPO-RUPO-bs${TRAIN_BATCH_SIZE}-n${ROLLOUT_N}-alpha${ALPHA}-lr${LR}-lora${LORA_RANK}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.train_files=[$TRAIN_DATA/train.parquet] \
data.val_files=[$EVAL_DATA/test.parquet] \
data.apply_chat_template_kwargs={enable_thinking:true} \
trainer.group_name=SDPO-rupo \
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
actor_rollout_ref.rollout.val_kwargs.n=8"

echo "----------------------------------------------------------------"
echo "Starting SDPO Rubric Training (Qwen3-30B-A3B-Thinking + LoRA)"
echo "Experiment: $EXP_NAME"
echo "Train data: $TRAIN_DATA (margin reward)"
echo "Eval data:  $EVAL_DATA (absolute reward)"
echo "Model: $MODEL_PATH"
echo "GPUs/node: $N_GPUS_PER_NODE  TP: $TP_SIZE  LoRA rank: $LORA_RANK"
echo "Judge: $JUDGE_MODEL"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$TRAIN_DATA" $ARGS
