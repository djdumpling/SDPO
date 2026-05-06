#!/bin/bash
# Retrain RuPO with GPT-5 (cross-family) as the training-time judge.
# Goal: falsify H5 (shared-model-family Goodhart). If self-judge accuracy under
# this run RECOVERS toward GPT-5's 0.77 baseline (instead of dropping like the
# Qwen3-8B-judge runs did), H5 is supported and the fix is "use a different-
# family judge." If self-judge stays low, H5 is wrong and another mechanism
# explains the cross-judge gap.
#
# Cost monitoring:
#   - Per train step: ~480 GPT-5 calls (12 prompts × 4 rollouts × ~5 criteria × 2 sides)
#   - Per val round: ~8.2k GPT-5 calls (819 prompts × ~10 calls)
#   - At ~$0.0075/call: ~$3.6 per train step, ~$60 per val round
#   - 5-step pilot (1 val): ~$80 total
#   - 50-step full (1 val): ~$240 total
#
# Saves only ONE checkpoint (the final), per user request. Each Qwen3-8B FSDP
# checkpoint is ~92 GB.

set -euo pipefail

OUT_DIR="${1:-${SDPO_NEXT_EXP_DIR:-}}"
if [[ -z "$OUT_DIR" ]]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi
mkdir -p "$OUT_DIR"

REPO_ROOT="/root/rubric-policy-optimization"
SDPO="$REPO_ROOT/SDPO"

# Load PRIME_API_KEY from .env for Pinference
set -a
source "$REPO_ROOT/.env"
set +a
# Activate venv
source "$SDPO/.venv/bin/activate"

if [[ -z "${PRIME_API_KEY:-}" ]]; then
  echo "ERROR: PRIME_API_KEY not set" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Judge endpoint: Pinference / GPT-5
# -----------------------------------------------------------------------------
# JUDGE_TIMEOUT=60 fail-fasts hung Pinference calls; the previous Phase 5 stalls
# were on indefinite hangs. JUDGE_MAX_CONCURRENT=8 keeps us under Pinference
# rate limits during the heavy training-time judging.
export OPENAI_BASE_URL="https://api.pinference.ai/api/v1"
export OPENAI_API_KEY="$PRIME_API_KEY"
export JUDGE_BASE_URL="https://api.pinference.ai/api/v1"
export JUDGE_MODEL="openai/gpt-5"
export JUDGE_ENABLE_THINKING="false"
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-8}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-60}"

# -----------------------------------------------------------------------------
# Training knobs (overridable by caller; sensible defaults match seed=2 baseline
# we measured in the diagnostic battery)
# -----------------------------------------------------------------------------
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
# Save final-only: SAVE_FREQ = TOTAL_TRAINING_STEPS, keep 1
export SAVE_FREQ="${SAVE_FREQ:-$TOTAL_TRAINING_STEPS}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export TEST_FREQ="${TEST_FREQ:-$TOTAL_TRAINING_STEPS}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export VAL_KWARGS_N="${VAL_KWARGS_N:-1}"

# Match seed=2 control run we mined for Phase 2
export SEED="${SEED:-2}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
export LR="${LR:-2e-6}"
export ALPHA="${ALPHA:-0.5}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export JUDGE_REWARD_TYPE_TRAIN="${JUDGE_REWARD_TYPE_TRAIN:-criteria_total_absolute}"
export JUDGE_REWARD_TYPE_VAL="${JUDGE_REWARD_TYPE_VAL:-criteria_total_absolute}"

# 6-GPU policy (the static judge GPUs are unused here; vLLM judge is GPT-5 API)
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-6}"
export NNODES="${NNODES:-1}"

# Dump rollouts so we can re-mine for GRPO degeneracy on the new judge
ROLLOUT_DIR="$OUT_DIR/rollout_dump"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$ROLLOUT_DIR}"
VAL_DIR="$OUT_DIR/val_generations"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$VAL_DIR}"

# Suffix that's recognizable in checkpoint paths and W&B
PILOT_TAG="${PILOT_TAG:-gpt5judge_$(date +%Y%m%d-%H%M%S)}"
SUFFIX="${PILOT_TAG}"

echo "----------------------------------------------------------------"
echo "GPT-5-judge retrain (H5 falsifier)"
echo "  steps=$TOTAL_TRAINING_STEPS  save_freq=$SAVE_FREQ  keep=$MAX_ACTOR_CKPT_TO_KEEP  test_freq=$TEST_FREQ"
echo "  judge_model=$JUDGE_MODEL  judge_base_url=$JUDGE_BASE_URL"
echo "  judge_max_concurrent=$JUDGE_MAX_CONCURRENT  judge_timeout=${JUDGE_TIMEOUT}s"
echo "  reward_type=$JUDGE_REWARD_TYPE_TRAIN  rollout_n=$ROLLOUT_N  seed=$SEED"
echo "  out_dir=$OUT_DIR"
echo "----------------------------------------------------------------"

bash "$SDPO/run_sdpo_qwen35_9b_litbench_8gpu.sh" "$SUFFIX"
