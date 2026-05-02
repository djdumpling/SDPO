#!/bin/bash
#SBATCH --job-name=sdpo-qwen35-9b-litbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=24:00:00
#SBATCH --output=/root/rubric-policy-optimization/SDPO/logs/%x-%j.out
#SBATCH --error=/root/rubric-policy-optimization/SDPO/logs/%x-%j.err

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/root/rubric-policy-optimization}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/SDPO/.venv}"
POLICY_MODEL_HF="${POLICY_MODEL_HF:-Qwen/Qwen3.5-9B}"
DATASET_NAME="${DATASET_NAME:-sumuks/litbench-ha-with-baseline}"
TRAIN_DATA="${TRAIN_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_absolute}"
# The judge is the same Qwen3.5-9B model served from a separate inference box.
# OPENAI_BASE_URL is only the OpenAI-compatible vLLM API URL, not an OpenAI model.
JUDGE_MODEL="${JUDGE_MODEL:-$POLICY_MODEL_HF}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${JUDGE_BASE_URL:-}}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing uv virtualenv at $VENV_PATH."
  echo "Create it on the training node before submitting, for example:"
  echo "  cd $REPO_ROOT/SDPO"
  echo "  uv sync"
  exit 1
fi

source "$REPO_ROOT/SDPO/.venv/bin/activate"

# Optional .env can carry WANDB_API_KEY, OPENAI_BASE_URL, JUDGE_MODEL, etc.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL:-${JUDGE_BASE_URL:-}}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-$POLICY_MODEL_HF}"

if [[ -z "$OPENAI_BASE_URL" ]]; then
  echo "Missing external judge endpoint."
  echo "Set OPENAI_BASE_URL, for example:"
  echo "  export OPENAI_BASE_URL=http://<judge-host>:8000/v1"
  echo "  export JUDGE_MODEL=$JUDGE_MODEL"
  exit 1
fi

# If no wandb key is present, run offline so training does not crash on auth.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY not set; using WANDB_MODE=offline."
  export WANDB_MODE=offline
fi

cd "$REPO_ROOT/SDPO"
mkdir -p logs

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

echo "=== Job env ==="
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "NODE=$(hostname)"
echo "VENV_PATH=$VENV_PATH"
echo "PYTHON=$(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF_HOME=$HF_HOME"
echo "POLICY_MODEL_HF=$POLICY_MODEL_HF"
echo "DATASET_NAME=$DATASET_NAME"
echo "TRAIN_DATA=$TRAIN_DATA"
echo "EVAL_DATA=$EVAL_DATA"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "JUDGE_MODEL=$JUDGE_MODEL"
if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "WANDB_API_KEY=SET"; else echo "WANDB_API_KEY=UNSET"; fi
echo "WANDB_ENTITY=${WANDB_ENTITY:-<default>}"
echo "WANDB_MODE=${WANDB_MODE:-online}"
echo "==============="

# Fail before allocating rollout workers if the Hugging Face rows have not been
# converted into the verl parquet schema used by sdpo_rupo.yaml.
if [[ ! -f "$TRAIN_DATA/train.parquet" || ! -f "$EVAL_DATA/test.parquet" ]]; then
  echo "Missing preprocessed LitBench parquet files."
  echo "Run these from $REPO_ROOT/SDPO before submitting:"
  echo "  uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \\"
  echo "      --dataset_name $DATASET_NAME --policy_prompt_mode pair --judge_reward_type margin --output_dir $TRAIN_DATA"
  echo "  uv run --active python -m environments.dpo_to_rupo_verl.preprocess_dataset \\"
  echo "      --dataset_name $DATASET_NAME --policy_prompt_mode pair --judge_reward_type absolute --output_dir $EVAL_DATA"
  exit 1
fi

echo "Checking external judge endpoint..."
if ! curl -s --max-time 10 "${OPENAI_BASE_URL%/}/models" > /dev/null; then
  echo "External judge endpoint did not answer at ${OPENAI_BASE_URL%/}/models."
  echo "Start the judge server first, and make sure OPENAI_BASE_URL includes /v1."
  exit 1
fi

nvidia-smi

export OPENAI_BASE_URL
export OPENAI_API_KEY
export JUDGE_MODEL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export N_GPUS_PER_NODE=8
export TP_SIZE="${TP_SIZE:-8}"
export MODEL_PATH="$POLICY_MODEL_HF"
export DATASET_NAME
export TRAIN_DATA
export EVAL_DATA

bash ./run_sdpo_qwen35_9b_litbench_8gpu.sh "${SLURM_JOB_ID:-manual}"
