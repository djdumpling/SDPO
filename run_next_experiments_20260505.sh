#!/bin/bash
# Tonight's experiments (post-overnight-diagnostics):
#   Track A — small fixes
#     A2: HS3 + Arena GPT-5 baselines (re-run last night's stalls with --judge-timeout 60)
#     A3: minimal-prompt LitBench baseline (re-run with thinking disabled + max_tokens=64)
#   Track B — H5 falsifier retrain (frozen judge → GPT-5 cross-family)
#     B0: 5-step pilot to verify plumbing. Reports cost-per-step actuals.
#     B1: NOT auto-run. Build only; user decides after seeing pilot results.

set -uo pipefail

REPO_ROOT="/root/rubric-policy-optimization"
SDPO="$REPO_ROOT/SDPO"
TS="$(date +%Y%m%d-%H%M%S)"
ROOT="$SDPO/next_experiments_${TS}"
mkdir -p "$ROOT"

cd "$SDPO"
set -a
source "$REPO_ROOT/.env"
set +a
# Activate venv so `python` resolves to one with openai/vllm/etc installed
source "$SDPO/.venv/bin/activate"

if [[ -z "${PRIME_API_KEY:-}" ]]; then
  echo "ERROR: PRIME_API_KEY not loaded from .env" >&2
  exit 1
fi
export PRIME_API_KEY

STATUS="$ROOT/_status.txt"
: > "$STATUS"
log_phase() { echo "$1 $2" | tee -a "$STATUS"; }

LITBENCH_BASE_JSONL="$SDPO/logs/untrained_baseline_litbench_20260504-235633/val_generations/0.jsonl"
HS3_BASE_JSONL="$SDPO/logs/untrained_baseline_helpsteer3_20260505-002852/val_generations/0.jsonl"
ARENA_BASE_JSONL="$SDPO/logs/untrained_baseline_arena_20260505-005528/val_generations/0.jsonl"
LITBENCH_TRAINED_JSONL="$SDPO/logs/seed2_constantlr_total_absolute_20260504-071351/val_generations/100.jsonl"

VLLM_LOG="$ROOT/p0_vllm.log"
VLLM_PORT=8765

start_vllm() {
  if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM already up on :$VLLM_PORT"
    return 0
  fi
  echo "Starting Qwen3-8B vLLM on GPU 0 :$VLLM_PORT (using venv vllm CLI)"
  # Match the proven-working pattern from run_overnight_diagnostics.sh:
  # use the venv's `vllm` binary (not `python -m`) and isolate CUDA_VISIBLE_DEVICES
  # in a subshell so the parent shell's env doesn't pollute it.
  (
    export CUDA_VISIBLE_DEVICES=0
    nohup "$SDPO/.venv/bin/vllm" serve Qwen/Qwen3-8B \
      --served-model-name Qwen/Qwen3-8B \
      --host 0.0.0.0 --port "$VLLM_PORT" \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
      --max-model-len 16384 \
      >"$VLLM_LOG" 2>&1 &
    echo "$!" > "$ROOT/_vllm.pid"
  )
  for i in $(seq 1 90); do
    if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready after $i checks"
      return 0
    fi
    sleep 4
  done
  echo "vLLM failed to start; tail of log:"
  tail -40 "$VLLM_LOG"
  return 1
}

stop_vllm() {
  if [[ -f "$ROOT/_vllm.pid" ]]; then
    pid="$(cat "$ROOT/_vllm.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping vLLM pid $pid"
      kill "$pid" 2>/dev/null || true
      sleep 5
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
}

trap stop_vllm EXIT

# -----------------------------------------------------------------------------
# A2 — HS3 + Arena GPT-5 baselines (in parallel; long-running, no GPU)
# -----------------------------------------------------------------------------
mkdir -p "$ROOT/pA2_gpt5_baselines"
GPT5_LIMIT="${GPT5_LIMIT:-300}"

run_gpt5_baseline() {
  local tag="$1" jsonl="$2"
  local out="$ROOT/pA2_gpt5_baselines/${tag}.json"
  local lg="$ROOT/pA2_${tag}.log"
  echo "[pA2:$tag] starting (limit=$GPT5_LIMIT, judge-timeout=60s)"
  python "$SDPO/scripts/cross_judge_rubric.py" \
    --source jsonl --jsonl "$jsonl" \
    --judge-base-url "https://api.pinference.ai/api/v1" \
    --judge-model "openai/gpt-5" \
    --judge-api-key-env "PRIME_API_KEY" \
    --judge-timeout 60 \
    --max-concurrent 16 \
    --limit "$GPT5_LIMIT" \
    --output "$out" \
    > "$lg" 2>&1
  local rc=$?
  log_phase "pA2_${tag}" "$rc"
}

# Fire A2 jobs into the background (they don't compete for GPU)
run_gpt5_baseline "hs3_baseline_gpt5" "$HS3_BASE_JSONL" &
PA2_HS3=$!
run_gpt5_baseline "arena_baseline_gpt5" "$ARENA_BASE_JSONL" &
PA2_ARENA=$!

# -----------------------------------------------------------------------------
# A3 — minimal-prompt LitBench baseline (re-run with thinking disabled)
# -----------------------------------------------------------------------------
start_vllm || { log_phase "p0_vllm" 1; exit 1; }
log_phase "p0_vllm" 0

mkdir -p "$ROOT/pA3_minimal_prompt"
echo "[pA3] starting minimal-prompt baseline"
python "$SDPO/scripts/diag_minimal_prompt_baseline.py" \
  --jsonl "$LITBENCH_TRAINED_JSONL" \
  --base-url "http://localhost:${VLLM_PORT}/v1" \
  --model "Qwen/Qwen3-8B" \
  --api-key-env "LOCAL_DUMMY_KEY" \
  --max-tokens 64 \
  --max-concurrent 16 \
  --limit 400 \
  --out "$ROOT/pA3_minimal_prompt/litbench_minimal.json" \
  > "$ROOT/pA3_minimal_prompt.log" 2>&1
log_phase "pA3_minimal_prompt" $?

# Stop our vLLM — Track B uses verl's own infra
stop_vllm
trap - EXIT

# -----------------------------------------------------------------------------
# B0 — GPT-5-judge retrain pilot (5 steps, all GPUs, ~$40)
# -----------------------------------------------------------------------------
echo
echo "Waiting on A2 background jobs before kicking off Track B..."
wait "$PA2_HS3" 2>/dev/null || true
wait "$PA2_ARENA" 2>/dev/null || true
echo "A2 jobs finished."

if [[ "${SKIP_TRACK_B:-0}" == "1" ]]; then
  echo "SKIP_TRACK_B=1 — stopping after Track A. Track B launcher exists at run_retrain_gpt5_judge.sh"
  log_phase "track_b" "skipped"
  exit 0
fi

PILOT_LOG="$ROOT/pB0_pilot_gpt5judge.log"
echo "[pB0] starting 5-step GPT-5-judge retrain pilot (final-only checkpoint, $PILOT_LOG)"
TOTAL_TRAINING_STEPS=5 \
  SAVE_FREQ=5 \
  MAX_ACTOR_CKPT_TO_KEEP=1 \
  TEST_FREQ=5 \
  VAL_BEFORE_TRAIN=False \
  PILOT_TAG="${TS}_gpt5judge_pilot" \
  bash "$SDPO/run_retrain_gpt5_judge.sh" "$ROOT" \
  > "$PILOT_LOG" 2>&1
log_phase "pB0_pilot" $?

echo
echo "===== run_next_experiments_${TS} done ====="
cat "$STATUS"
echo
echo "Outputs at: $ROOT"
