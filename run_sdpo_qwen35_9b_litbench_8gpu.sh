#!/bin/bash

# Train SDPO rubric generation with Qwen dense models on LitBench preference pairs.
#
# Usage:
#   # 1. Preprocess LitBench twice so training and eval can use different reward shaping.
#   ./prepare_litbench_ha_with_baseline.sh
#
#   # 2. Submit the Slurm wrapper. It starts a local vLLM judge on two GPUs
#   #    and trains on the other six GPUs.
#
#   # 3. Run training from this directory or submit:
#   #      the 8xH200 Slurm wrapper.

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Keep model, dataset, and parquet paths overridable so the same launcher can be
# reused after changing judge models or LitBench variants.
CONFIG_NAME="sdpo_rupo"
DATASET_NAME="${DATASET_NAME:-sumuks/litbench-ha-with-baseline}"

# Reward type controls the gradient signal shape AND the parquet folder we read
# from. Train and val should match — the previous train=margin / val=absolute
# setup caused specification gaming (rubric-strictness inflation). criteria_margin
# is the recommended default: per-criterion gradient signal, forces structured
# <criterion> rubric format, less hackable than the whole-rubric margin reward.
# Run prepare_litbench_ha_with_baseline.sh with matching env vars before training.
JUDGE_REWARD_TYPE_TRAIN="${JUDGE_REWARD_TYPE_TRAIN:-criteria_margin}"
JUDGE_REWARD_TYPE_VAL="${JUDGE_REWARD_TYPE_VAL:-criteria_margin}"
TRAIN_DATA="${TRAIN_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_${JUDGE_REWARD_TYPE_TRAIN}}"
EVAL_DATA="${EVAL_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_${JUDGE_REWARD_TYPE_VAL}}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

# Non-thinking rubric generation should emit final XML directly. Batch 12 keeps
# the effective rollout count at 48 samples per step on six train GPUs.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
ROLLOUT_N=${ROLLOUT_N:-4}
LR=${LR:-2e-6}
ALPHA=${ALPHA:-0.5}

# Disable Qwen thinking mode for the policy path. The model should produce the
# public <analysis> block and <rubric> block without hidden thinking tokens.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_REPROMPT_LEN=${MAX_REPROMPT_LEN:-12288}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-12288}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
# `dummy` (verl default) inits vLLM with random weights and expects a runtime
# FSDP -> vLLM weight sync. That sync was not populating the rollout engine in
# our setup (rollouts emitted multilingual gibberish; rollout_log_ppl pinned at
# ln(vocab_size)). Force a real disk load so vLLM has correct base weights from
# step 0. `safetensors` is more explicit than `auto` and matches Qwen3-8B's
# on-disk format.
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}

# Checkpoint cadence. Each Qwen3-8B FSDP checkpoint is ~92 GB on disk (model
# shards + Adam optimizer state). With save_freq=50 over 200 steps that would
# write 4 * 92 GB = 368 GB and quickly exhaust the shared cluster filesystem.
# Default is now -1 (verl's "never save"). Override with e.g. SAVE_FREQ=200 to
# save only the final checkpoint, or SAVE_FREQ=50 to restore the old cadence.
# verl's checkpoint format always bundles model + optimizer; if you want a
# deployable model only, set MAX_ACTOR_CKPT_TO_KEEP=1 and strip optim shards
# afterward (~62 GB savings per kept checkpoint).
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-50}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-12}
# Rollouts per validation prompt. The val set has 819 prompts which already
# averages out per-prompt noise; extra samples just slow validation. n=1 keeps
# each validation run ~10 min instead of ~60 min for n=8.
VAL_KWARGS_N=${VAL_KWARGS_N:-1}
# Cap total optimizer steps. ppo_trainer.yaml defaults total_epochs=30 which
# would push the step count to 30 * (14924 / TRAIN_BATCH_SIZE). The first run
# (step-100 val=0.681) showed the policy saturating by step ~50 against the
# Qwen3-8B self-judge ceiling, so 200 steps is plenty for this configuration.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
# Bound checkpoint disk usage. Each Qwen3-8B full-param checkpoint is ~80 GB
# (actor weights + Adam state). With save_freq=50 over 3k steps that would be
# ~4.8 TB. Keep only the most recent N to cap at ~5 * 80 GB = 400 GB.
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-5}
# verl's default checkpoint path is checkpoints/${project_name}/${experiment_name}.
# ppo_trainer.yaml defaults are "verl_examples/gsm8k", which is misleading for
# LitBench runs. Override both so checkpoints land in a recognizable folder.
PROJECT_NAME=${PROJECT_NAME:-SDPO-RUPO-litbench}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}

# LORA_RANK=0 disables PEFT entirely and trains all 8B parameters via FSDP
# sharding. This is the default because vLLM 0.20.0 + --enable_lora produced
# corrupted forward passes (gibberish rollouts, log_ppl pinned at ln(vocab)).
# Full-parameter training fits in the 6x H200 budget and is what produced the
# step-100 val/reward/mean=0.681 baseline. Set LORA_RANK=32 to opt back into
# LoRA only after confirming the vLLM LoRA path is fixed.
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-32}

# Keep tensor parallelism off by default for the 8B/9B dense model on H200s so
# six rollout replicas can share the non-thinking rollout workload.
TP_SIZE=${TP_SIZE:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-6}
export NNODES=${NNODES:-1}

# The reward function talks to the OpenAI-compatible vLLM judge endpoint. The
# Slurm wrapper starts this locally before invoking this training launcher.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${JUDGE_BASE_URL:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_PATH}"
export JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-false}"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-32}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-900}"
export RUBRIC_FORMAT_REWARD_MAX="${RUBRIC_FORMAT_REWARD_MAX:-0.0}"

SUFFIX=${1:-"qwen3_8b_litbench_ha_with_baseline_local_judge_nonthinking"}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Resolve the SDPO root once so all paths below are stable no matter where the
# user launches the script from.
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

if [[ -z "$OPENAI_BASE_URL" ]]; then
    echo "Missing judge endpoint."
    echo "Run through the 8xH200 Slurm wrapper,"
    echo "or export OPENAI_BASE_URL before invoking this helper directly."
    exit 1
fi

# Fail before allocating rollouts if the Hugging Face dataset has not been
# converted into the verl parquet schema expected by sdpo_rupo.yaml.
if [[ ! -f "$PROJECT_ROOT/$TRAIN_DATA/train.parquet" || ! -f "$PROJECT_ROOT/$EVAL_DATA/test.parquet" ]]; then
    echo "Missing preprocessed LitBench parquet files."
    echo "Run these from $PROJECT_ROOT:"
    echo "  ./prepare_litbench_ha_with_baseline.sh"
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
data.apply_chat_template_kwargs={enable_thinking:$ENABLE_THINKING} \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
trainer.group_name=SDPO-rupo \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=$NNODES \
trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXP_NAME \
trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
data.val_batch_size=$VAL_BATCH_SIZE \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
actor_rollout_ref.rollout.load_format=$ROLLOUT_LOAD_FORMAT \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.lora_rank=$LORA_RANK \
actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_REPROMPT_LEN \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_KWARGS_N \
+ray_kwargs.ray_init.include_dashboard=False"

echo "----------------------------------------------------------------"
echo "Starting SDPO Rubric Training (Qwen dense + LoRA)"
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET_NAME"
echo "Train data: $TRAIN_DATA (reward type: $JUDGE_REWARD_TYPE_TRAIN)"
echo "Eval data:  $EVAL_DATA (reward type: $JUDGE_REWARD_TYPE_VAL)"
echo "Model: $MODEL_PATH"
echo "GPUs/node: $N_GPUS_PER_NODE  TP: $TP_SIZE  LoRA rank: $LORA_RANK"
echo "Thinking: $ENABLE_THINKING  response length: $MAX_RESPONSE_LENGTH  rollout max model len: $ROLLOUT_MAX_MODEL_LEN"
echo "Rollout KV: gpu util $ROLLOUT_GPU_MEMORY_UTILIZATION  max batched tokens $ROLLOUT_MAX_NUM_BATCHED_TOKENS  max seqs $ROLLOUT_MAX_NUM_SEQS"
echo "Judge endpoint: $OPENAI_BASE_URL"
echo "Judge model: $JUDGE_MODEL  judge thinking: $JUDGE_ENABLE_THINKING"
echo "Format reward max: $RUBRIC_FORMAT_REWARD_MAX  rollout dump: $ROLLOUT_DATA_DIR"
echo "Rollout load format: $ROLLOUT_LOAD_FORMAT"
echo "Save freq: $SAVE_FREQ  test freq: $TEST_FREQ  val batch size: $VAL_BATCH_SIZE  val n: $VAL_KWARGS_N"
echo "Total training steps: $TOTAL_TRAINING_STEPS"
if [[ "$SAVE_FREQ" -le 0 ]]; then
  echo "Checkpoints: DISABLED (set SAVE_FREQ>0 to enable; each save is ~92 GB)"
else
  echo "Checkpoint dir: $PROJECT_ROOT/checkpoints/$PROJECT_NAME/$EXP_NAME (keeping last $MAX_ACTOR_CKPT_TO_KEEP)"
fi
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$TRAIN_DATA" $ARGS
