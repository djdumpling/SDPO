#!/bin/bash
#SBATCH --job-name=sdpo-qwen35-9b-litbench
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/root/rubric-policy-optimization/SDPO/logs/%x-%j.out
#SBATCH --error=/root/rubric-policy-optimization/SDPO/logs/%x-%j.err

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/root/rubric-policy-optimization}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/SDPO/.venv}"
POLICY_MODEL_HF="${POLICY_MODEL_HF:-Qwen/Qwen3-8B}"
DATASET_NAME="${DATASET_NAME:-sumuks/litbench-ha-with-baseline}"
TRAIN_DATA="${TRAIN_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_absolute}"
JUDGE_MODEL_HF="${JUDGE_MODEL_HF:-$POLICY_MODEL_HF}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-2}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.55}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-false}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing uv virtualenv at $VENV_PATH."
  echo "Install SDPO dependencies on the training node before submitting."
  echo "One known-good path is the repo's SDPO install instructions, then create this uv environment:"
  echo "  cd $REPO_ROOT/SDPO"
  echo "  uv venv .venv"
  echo "  source .venv/bin/activate"
  exit 1
fi

source "$REPO_ROOT/SDPO/.venv/bin/activate"

# Optional .env can carry WANDB_API_KEY, JUDGE_MAX_CONCURRENT, model overrides, etc.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
JUDGE_MODEL_HF="${JUDGE_MODEL_HF:-$POLICY_MODEL_HF}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-2}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.55}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
JUDGE_ENABLE_THINKING="${JUDGE_ENABLE_THINKING:-false}"

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
echo "JUDGE_MODEL_HF=$JUDGE_MODEL_HF"
echo "JUDGE_PORT=$JUDGE_PORT"
echo "JUDGE_TP_SIZE=$JUDGE_TP_SIZE"
echo "JUDGE_GPU_MEMORY_UTILIZATION=$JUDGE_GPU_MEMORY_UTILIZATION"
echo "JUDGE_MAX_MODEL_LEN=$JUDGE_MAX_MODEL_LEN"
echo "JUDGE_ENABLE_THINKING=$JUDGE_ENABLE_THINKING"
echo "DATASET_NAME=$DATASET_NAME"
echo "TRAIN_DATA=$TRAIN_DATA"
echo "EVAL_DATA=$EVAL_DATA"
if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "WANDB_API_KEY=SET"; else echo "WANDB_API_KEY=UNSET"; fi
echo "WANDB_ENTITY=${WANDB_ENTITY:-<default>}"
echo "WANDB_MODE=${WANDB_MODE:-online}"
echo "==============="

# Fail before allocating rollout workers if the Hugging Face rows have not been
# converted into the verl parquet schema used by sdpo_rupo.yaml.
if [[ ! -f "$TRAIN_DATA/train.parquet" || ! -f "$EVAL_DATA/test.parquet" ]]; then
  echo "Missing preprocessed LitBench parquet files."
  echo "Run these from $REPO_ROOT/SDPO before submitting:"
  echo "  ./prepare_litbench_ha_with_baseline.sh"
  exit 1
fi

nvidia-smi

# Judge: OpenAI-compatible vLLM server on GPU indexes 0 and 1.
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL_HF" \
  --served-model-name "$JUDGE_MODEL_HF" \
  --host 127.0.0.1 \
  --port "$JUDGE_PORT" \
  --tensor-parallel-size "$JUDGE_TP_SIZE" \
  --gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$JUDGE_MAX_MODEL_LEN" \
  > "logs/judge-${SLURM_JOB_ID:-manual}.log" 2>&1 &
JUDGE_PID=$!

cleanup() {
  echo "Stopping judge (pid=$JUDGE_PID)..."
  kill "$JUDGE_PID" 2>/dev/null || true
  wait "$JUDGE_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for judge server on port $JUDGE_PORT..."
for i in $(seq 1 120); do
  if curl -s --max-time 2 "http://127.0.0.1:${JUDGE_PORT}/v1/models" > /dev/null; then
    echo "Judge ready after ${i} attempts."
    break
  fi
  if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
    echo "Judge process died before becoming ready. See logs/judge-${SLURM_JOB_ID:-manual}.log"
    exit 1
  fi
  sleep 5
done

if ! curl -s --max-time 2 "http://127.0.0.1:${JUDGE_PORT}/v1/models" > /dev/null; then
  echo "Judge failed to come up within 10 minutes."
  exit 1
fi

export OPENAI_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export OPENAI_API_KEY
export JUDGE_MODEL="$JUDGE_MODEL_HF"
export JUDGE_ENABLE_THINKING
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export N_GPUS_PER_NODE=6
export TP_SIZE="${TP_SIZE:-1}"
export MODEL_PATH="$POLICY_MODEL_HF"
export DATASET_NAME
export TRAIN_DATA
export EVAL_DATA

bash ./run_sdpo_qwen35_9b_litbench_8gpu.sh "${SLURM_JOB_ID:-manual}"
