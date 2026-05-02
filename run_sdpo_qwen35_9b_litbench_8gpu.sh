#!/bin/bash

# Train SDPO rubric generation with Qwen3.5-9B on LitBench preference pairs.
#
# Usage:
#   # 1. Preprocess LitBench twice so training and eval can use different reward shaping.
#   uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \
#       --dataset_name sumuks/litbench-ha-with-baseline --policy_prompt_mode pair \
#       --judge_reward_type margin --output_dir datasets/dpo_to_rupo_litbench_ha_with_baseline_margin
#   uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \
#       --dataset_name sumuks/litbench-ha-with-baseline --policy_prompt_mode pair \
#       --judge_reward_type absolute --output_dir datasets/dpo_to_rupo_litbench_ha_with_baseline_absolute
#
#   # 2. Start a vLLM OpenAI-compatible Qwen3.5-9B judge endpoint elsewhere and export:
#   #      OPENAI_BASE_URL=http://<judge-host>:8000/v1
#   #      JUDGE_MODEL=Qwen/Qwen3.5-9B
#
#   # 3. Run training from this directory or submit:
#   #      train_qwen35_9b_litbench_8xh200_external_judge.slurm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Keep model, dataset, and parquet paths overridable so the same launcher can be
# reused after changing judge models or LitBench variants.
CONFIG_NAME="sdpo_rupo"
DATASET_NAME="${DATASET_NAME:-sumuks/litbench-ha-with-baseline}"
TRAIN_DATA="${TRAIN_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_absolute}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"

# Conservative defaults for 9B dense on 8 H200s. Increase train batch only after
# a short smoke run confirms memory and judge throughput are stable.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
LR=${LR:-5e-6}
ALPHA=${ALPHA:-0.5}

# LoRA keeps the full-node run memory-predictable while still letting all eight
# GPUs participate in rollout and actor work.
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

# Use all eight training GPUs by default. Override TP_SIZE only if the cluster
# launcher or model engine requires a different rollout tensor-parallel shape.
TP_SIZE=${TP_SIZE:-8}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}

# The reward function talks to an external vLLM endpoint serving Qwen3.5-9B.
# OPENAI_BASE_URL is the API compatibility variable expected by the OpenAI client.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${JUDGE_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_PATH}"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-64}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-900}"

SUFFIX=${1:-"qwen35_9b_litbench_ha_with_baseline_external_judge"}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Resolve the SDPO root once so all paths below are stable no matter where the
# user launches the script from.
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

if [[ -z "$OPENAI_BASE_URL" ]]; then
    echo "Missing external judge endpoint."
    echo "Set OPENAI_BASE_URL, for example:"
    echo "  export OPENAI_BASE_URL=http://<judge-host>:8000/v1"
    echo "  export JUDGE_MODEL=$JUDGE_MODEL"
    exit 1
fi

# Fail before allocating rollouts if the Hugging Face dataset has not been
# converted into the verl parquet schema expected by sdpo_rupo.yaml.
if [[ ! -f "$PROJECT_ROOT/$TRAIN_DATA/train.parquet" || ! -f "$PROJECT_ROOT/$EVAL_DATA/test.parquet" ]]; then
    echo "Missing preprocessed LitBench parquet files."
    echo "Run these from $PROJECT_ROOT:"
    echo "  uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \\"
    echo "      --dataset_name $DATASET_NAME --policy_prompt_mode pair --judge_reward_type margin --output_dir $TRAIN_DATA"
    echo "  uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \\"
    echo "      --dataset_name $DATASET_NAME --policy_prompt_mode pair --judge_reward_type absolute --output_dir $EVAL_DATA"
    exit 1
fi

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

# Include the model and endpoint-backed setup in the experiment name so W&B and
# checkpoint paths remain interpretable after several LitBench runs.
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="SDPO-RUPO-bs${TRAIN_BATCH_SIZE}-n${ROLLOUT_N}-alpha${ALPHA}-lr${LR}-lora${LORA_RANK}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.train_files=[$TRAIN_DATA/train.parquet] \
data.val_files=[$EVAL_DATA/test.parquet] \
data.apply_chat_template_kwargs={enable_thinking:false} \
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
actor_rollout_ref.rollout.val_kwargs.n=8 \
+ray_kwargs.ray_init.include_dashboard=False"

echo "----------------------------------------------------------------"
echo "Starting SDPO Rubric Training (Qwen3.5-9B + LoRA)"
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET_NAME"
echo "Train data: $TRAIN_DATA (margin reward)"
echo "Eval data:  $EVAL_DATA (absolute reward)"
echo "Model: $MODEL_PATH"
echo "GPUs/node: $N_GPUS_PER_NODE  TP: $TP_SIZE  LoRA rank: $LORA_RANK"
echo "Judge endpoint: $OPENAI_BASE_URL"
echo "Judge model: $JUDGE_MODEL"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$TRAIN_DATA" $ARGS
