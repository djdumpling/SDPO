#!/bin/bash
#SBATCH --job-name=sdpo-qwen4b-litbench
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=640G
#SBATCH --time=24:00:00
#SBATCH --output=/home/aw2295/scratch_pi_btk22/aw2295/sdpo/rubric-policy-optimization/SDPO/logs/%x-%j.out
#SBATCH --error=/home/aw2295/scratch_pi_btk22/aw2295/sdpo/rubric-policy-optimization/SDPO/logs/%x-%j.err

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/home/aw2295/scratch_pi_btk22/aw2295/sdpo/rubric-policy-optimization}"
CONDA_ENV="${CONDA_ENV:-sdpo}"

JUDGE_MODEL_HF="${JUDGE_MODEL_HF:-Qwen/Qwen3-4B-Instruct-2507}"
JUDGE_PORT=8000
POLICY_MODEL_HF="${POLICY_MODEL_HF:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET_NAME="${DATASET_NAME:-sumuks/litbench-ha-with-baseline}"
TRAIN_DATA="${TRAIN_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_margin}"
EVAL_DATA="${EVAL_DATA:-datasets/dpo_to_rupo_litbench_ha_with_baseline_absolute}"

# Load conda via Bouchet's Lmod.
source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile 2>/dev/null || true
module load miniconda
conda activate "$CONDA_ENV"

# Optional .env (wandb key, judge overrides, etc.).
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

# If no wandb key, run offline so training doesn't crash on auth.
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
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF_HOME=$HF_HOME"
echo "POLICY_MODEL_HF=$POLICY_MODEL_HF"
echo "DATASET_NAME=$DATASET_NAME"
echo "TRAIN_DATA=$TRAIN_DATA"
echo "EVAL_DATA=$EVAL_DATA"
if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "WANDB_API_KEY=SET"; else echo "WANDB_API_KEY=UNSET"; fi
echo "WANDB_ENTITY=${WANDB_ENTITY:-<default>}"
echo "WANDB_MODE=${WANDB_MODE:-online}"
echo "==============="

# Fail before starting the judge if the Hugging Face rows have not been
# converted into the verl parquet schema used by sdpo_rupo.yaml.
if [[ ! -f "$TRAIN_DATA/train.parquet" || ! -f "$EVAL_DATA/test.parquet" ]]; then
  echo "Missing preprocessed LitBench parquet files."
  echo "Run these from $REPO_ROOT/SDPO before submitting:"
  echo "  uv run --active --project .. python -m environments.dpo_to_rupo_verl.preprocess_dataset \\"
  echo "      --dataset_name $DATASET_NAME --policy_prompt_mode pair --judge_reward_type margin --output_dir $TRAIN_DATA"
  echo "  uv run --active --project .. python -m environments.dpo_to_rupo_verl.preprocess_dataset \\"
  echo "      --dataset_name $DATASET_NAME --policy_prompt_mode pair --judge_reward_type absolute --output_dir $EVAL_DATA"
  exit 1
fi

nvidia-smi

# Judge: Qwen3-4B on GPU index 4.
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL_HF" \
  --host 127.0.0.1 \
  --port "$JUDGE_PORT" \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
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
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export JUDGE_MODEL="$JUDGE_MODEL_HF"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS_PER_NODE=4
export TP_SIZE=4
export MODEL_PATH="$POLICY_MODEL_HF"
export DATASET_NAME
export TRAIN_DATA
export EVAL_DATA

bash ./run_sdpo_qwen4b_litbench.sh "${SLURM_JOB_ID:-manual}"
