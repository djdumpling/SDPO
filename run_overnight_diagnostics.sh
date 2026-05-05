#!/usr/bin/env bash
# Overnight diagnostic battery for the RuPO/SDPO trained-vs-untrained parity finding.
# Implements steps from 05_04_26_research_plan.md (the runnable subset — Steps 4/5/11
# require trained checkpoints that weren't saved, so they're skipped).
#
# Phases (each writes to $DIAG_ROOT/<phase>/):
#   0 — start local Qwen3-8B vLLM (1 H200) for judge calls
#   1 — static audits: train-vs-val prompt mode
#   2 — GRPO group-degeneracy from train-time rollout dump
#   3 — judge test-retest k=5 on 100 fixed triples (H1: judge SNR)
#   4 — judge ensembling k=4 on existing val rubrics (H1: noise-floor lift)
#   5 — GPT-5 cross-judge on UNTRAINED baselines (fills the missing GPT-5 trained-vs-base table)
#   6 — position-shuffle re-eval on LitBench (H7)
#   7 — minimal-prompt baseline (H11: prompt-template ceiling)
#   8 — write SUMMARY.md
#
# Usage:
#   PRIME_API_KEY=... bash run_overnight_diagnostics.sh
#
# Optional env knobs: DIAG_ROOT, LOCAL_VLLM_PORT, GPT5_LIMIT, SKIP_GPT5=1, etc.

set -uo pipefail   # NOT -e: each phase tracks its own rc; we keep going

# Source repo .env so PRIME_API_KEY, etc. are visible
REPO_ROOT_FOR_ENV="/root/rubric-policy-optimization"
if [[ -f "$REPO_ROOT_FOR_ENV/.env" ]]; then
  set -a; source "$REPO_ROOT_FOR_ENV/.env"; set +a
fi

# ---------- config ----------
TS=$(date '+%Y%m%d-%H%M%S')
DIAG_ROOT="${DIAG_ROOT:-/root/rubric-policy-optimization/SDPO/diagnostics_${TS}}"
mkdir -p "$DIAG_ROOT"

REPO_ROOT="/root/rubric-policy-optimization"
SDPO_ROOT="$REPO_ROOT/SDPO"
SCRIPTS="$SDPO_ROOT/scripts"
PYTHON="$SDPO_ROOT/.venv/bin/python"
VLLM_BIN="$SDPO_ROOT/.venv/bin/vllm"

# Existing rubric sources (output of prior verl validation_data_dir dumps)
LITBENCH_TRAINED_VAL="$SDPO_ROOT/logs/seed2_constantlr_total_absolute_20260504-071351/val_generations/100.jsonl"
LITBENCH_BASELINE_VAL="$SDPO_ROOT/logs/untrained_baseline_litbench_20260504-235633/val_generations/0.jsonl"
HS3_TRAINED_VAL="$SDPO_ROOT/logs/helpsteer3_total_absolute_v2_20260504-181504/val_generations/100.jsonl"
HS3_BASELINE_VAL="$SDPO_ROOT/logs/untrained_baseline_helpsteer3_20260505-002852/val_generations/0.jsonl"
ARENA_TRAINED_VAL="$SDPO_ROOT/logs/arena_total_absolute_v3_20260504-213320/val_generations/100.jsonl"
ARENA_BASELINE_VAL="$SDPO_ROOT/logs/untrained_baseline_arena_20260505-005528/val_generations/0.jsonl"

# Train-time rollout dump (for Phase 2 GRPO group analysis)
TRAIN_ROLLOUT_DIR="/logs/seed2_constantlr_total_absolute_20260504-071351_rollout_dump/run-20260504-071432-criteria_total_absolute"

# Local vLLM serving config
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-8765}"   # avoid collisions
LOCAL_VLLM_GPU="${LOCAL_VLLM_GPU:-0}"
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-Qwen/Qwen3-8B}"
LOCAL_JUDGE_URL="http://localhost:${LOCAL_VLLM_PORT}/v1"
export LOCAL_DUMMY_KEY="${LOCAL_DUMMY_KEY:-no-key-needed}"

# GPT-5 endpoint
GPT5_BASE_URL="${GPT5_BASE_URL:-https://api.pinference.ai/api/v1}"
GPT5_MODEL="${GPT5_MODEL:-openai/gpt-5}"
GPT5_LIMIT="${GPT5_LIMIT:-400}"   # n per dataset; baseline files are 819/500/940 full

# Test-retest knobs
TR_N_TRIPLES="${TR_N_TRIPLES:-100}"
TR_K_RESAMPLES="${TR_K_RESAMPLES:-5}"

# Ensembling knobs
ENS_K="${ENS_K:-4}"
ENS_LIMIT="${ENS_LIMIT:-300}"   # subsample to keep local runtime ~30 min/run

# Skip flags (for re-runs)
SKIP_VLLM="${SKIP_VLLM:-0}"      # set to 1 if you've already started vLLM externally
SKIP_GPT5="${SKIP_GPT5:-0}"      # set to 1 to skip the paid phase

# ---------- helpers ----------
log_phase() {
  echo ""
  echo "================================================================"
  echo "[$(date '+%H:%M:%S')]  $1"
  echo "================================================================"
}

run_phase() {
  local name="$1"; shift
  local logfile="$DIAG_ROOT/${name}.log"
  log_phase "$name"
  echo "  log: $logfile"
  ( "$@" ) > "$logfile" 2>&1
  local rc=$?
  echo "  rc=$rc"
  echo "$name $rc" >> "$DIAG_ROOT/_status.txt"
  return $rc
}

wait_for_endpoint() {
  local url="$1"; local max_s="${2:-600}"
  local start=$SECONDS
  echo "  waiting up to ${max_s}s for $url ..."
  while (( SECONDS - start < max_s )); do
    if curl -sf "$url/models" >/dev/null 2>&1; then
      echo "  endpoint ready after $((SECONDS-start))s"
      return 0
    fi
    sleep 5
  done
  echo "  TIMEOUT waiting for $url"
  return 1
}

# ---------- Phase 0: start local vLLM (1 H200) ----------
VLLM_PID=""
if [[ "$SKIP_VLLM" == "1" ]]; then
  log_phase "Phase 0 — vLLM (skipped, using external endpoint $LOCAL_JUDGE_URL)"
else
  log_phase "Phase 0 — starting local vLLM serving Qwen3-8B on GPU $LOCAL_VLLM_GPU port $LOCAL_VLLM_PORT"
  CUDA_VISIBLE_DEVICES="$LOCAL_VLLM_GPU" \
    "$VLLM_BIN" serve "$LOCAL_JUDGE_MODEL" \
    --tensor-parallel-size 1 \
    --port "$LOCAL_VLLM_PORT" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --served-model-name "$LOCAL_JUDGE_MODEL" \
    > "$DIAG_ROOT/p0_vllm.log" 2>&1 &
  VLLM_PID=$!
  echo "  vllm pid=$VLLM_PID  log=$DIAG_ROOT/p0_vllm.log"
  if ! wait_for_endpoint "$LOCAL_JUDGE_URL" 900; then
    echo "  vLLM did not start; aborting"
    [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
    exit 1
  fi
fi

# Cleanup hook
cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    echo "  stopping vllm pid=$VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---------- Phase 1: static audits ----------
mkdir -p "$DIAG_ROOT/p1_audits"
run_phase "p1_prompt_mode_audit" \
  "$PYTHON" "$SCRIPTS/diag_prompt_mode_audit.py" \
    --launcher "$SDPO_ROOT/run_sdpo_qwen35_9b_litbench_8gpu.sh" \
    --preprocess "$SDPO_ROOT/environments/dpo_to_rupo_verl/preprocess_dataset.py" \
    --baseline-prep "$SDPO_ROOT/prepare_litbench_ha_with_baseline.sh" \
    --out "$DIAG_ROOT/p1_audits/prompt_mode.json"

# ---------- Phase 2: GRPO group-degeneracy ----------
run_phase "p2_grpo_group_degeneracy" \
  "$PYTHON" "$SCRIPTS/diag_grpo_group_degeneracy.py" \
    --rollout-dir "$TRAIN_ROLLOUT_DIR" \
    --rollouts-per-prompt 4 \
    --out "$DIAG_ROOT/p2_grpo_group_degeneracy.json"

# ---------- Phase 3: judge test-retest k=5 ----------
run_phase "p3_judge_test_retest" \
  "$PYTHON" "$SCRIPTS/diag_judge_test_retest.py" \
    --jsonl "$LITBENCH_TRAINED_VAL" \
    --judge-base-url "$LOCAL_JUDGE_URL" \
    --judge-model "$LOCAL_JUDGE_MODEL" \
    --judge-api-key-env LOCAL_DUMMY_KEY \
    --n-triples "$TR_N_TRIPLES" \
    --k-resamples "$TR_K_RESAMPLES" \
    --temperature 0.7 \
    --max-concurrent 8 \
    --out "$DIAG_ROOT/p3_judge_test_retest.json"

# ---------- Phase 4: judge ensembling k=$ENS_K (LitBench trained + baseline) ----------
mkdir -p "$DIAG_ROOT/p4_ensembling"
for SOURCE_NAME in "litbench_trained:$LITBENCH_TRAINED_VAL" "litbench_baseline:$LITBENCH_BASELINE_VAL"; do
  NAME="${SOURCE_NAME%%:*}"
  PATH_VAL="${SOURCE_NAME##*:}"
  if [[ ! -f "$PATH_VAL" ]]; then
    echo "  skipping $NAME — file not found at $PATH_VAL"
    continue
  fi
  for KK in $(seq 1 "$ENS_K"); do
    run_phase "p4_${NAME}_k${KK}" \
      "$PYTHON" "$SCRIPTS/cross_judge_rubric.py" \
        --source jsonl --jsonl "$PATH_VAL" \
        --judge-base-url "$LOCAL_JUDGE_URL" \
        --judge-model "$LOCAL_JUDGE_MODEL" \
        --judge-api-key-env LOCAL_DUMMY_KEY \
        --temperature 0.7 \
        --max-concurrent 8 \
        --judge-timeout 300 \
        --limit "$ENS_LIMIT" \
        --output "$DIAG_ROOT/p4_ensembling/${NAME}_k${KK}.json"
  done
  run_phase "p4_${NAME}_aggregate" \
    "$PYTHON" "$SCRIPTS/diag_aggregate_ensemble.py" \
      --inputs "$DIAG_ROOT/p4_ensembling/${NAME}_k"*.json \
      --out "$DIAG_ROOT/p4_ensembling/${NAME}_ensembled.json"
done

# ---------- Phase 5: GPT-5 baselines (paid) ----------
if [[ "$SKIP_GPT5" == "1" ]]; then
  log_phase "Phase 5 — SKIPPED (SKIP_GPT5=1)"
else
  if [[ -z "${PRIME_API_KEY:-}" ]]; then
    echo "  PRIME_API_KEY not set — skipping Phase 5"
  else
    mkdir -p "$DIAG_ROOT/p5_gpt5_baselines"
    for SOURCE_NAME in "litbench_baseline:$LITBENCH_BASELINE_VAL" "hs3_baseline:$HS3_BASELINE_VAL" "arena_baseline:$ARENA_BASELINE_VAL"; do
      NAME="${SOURCE_NAME%%:*}"
      PATH_VAL="${SOURCE_NAME##*:}"
      [[ -f "$PATH_VAL" ]] || { echo "  skip $NAME (no file)"; continue; }
      run_phase "p5_gpt5_${NAME}" \
        "$PYTHON" "$SCRIPTS/cross_judge_rubric.py" \
          --source jsonl --jsonl "$PATH_VAL" \
          --judge-base-url "$GPT5_BASE_URL" \
          --judge-model "$GPT5_MODEL" \
          --judge-api-key-env PRIME_API_KEY \
          --max-concurrent 16 \
          --limit "$GPT5_LIMIT" \
          --output "$DIAG_ROOT/p5_gpt5_baselines/${NAME}_gpt5.json"
    done
  fi
fi

# ---------- Phase 6: position-shuffle (LitBench, local judge) ----------
mkdir -p "$DIAG_ROOT/p6_position_shuffle"
run_phase "p6_litbench_trained_flipped" \
  "$PYTHON" "$SCRIPTS/diag_position_shuffle_eval.py" \
    --jsonl "$LITBENCH_TRAINED_VAL" \
    --judge-base-url "$LOCAL_JUDGE_URL" \
    --judge-model "$LOCAL_JUDGE_MODEL" \
    --judge-api-key-env LOCAL_DUMMY_KEY \
    --max-concurrent 8 \
    --limit "$ENS_LIMIT" \
    --out "$DIAG_ROOT/p6_position_shuffle/litbench_trained_flipped.json"

# ---------- Phase 7: minimal-prompt baseline ----------
mkdir -p "$DIAG_ROOT/p7_minimal_prompt"
run_phase "p7_minimal_prompt_litbench" \
  "$PYTHON" "$SCRIPTS/diag_minimal_prompt_baseline.py" \
    --jsonl "$LITBENCH_BASELINE_VAL" \
    --base-url "$LOCAL_JUDGE_URL" \
    --model "$LOCAL_JUDGE_MODEL" \
    --api-key-env LOCAL_DUMMY_KEY \
    --max-concurrent 8 \
    --limit 400 \
    --out "$DIAG_ROOT/p7_minimal_prompt/litbench_minimal.json"

# ---------- Phase 8: SUMMARY ----------
run_phase "p8_summarize" \
  "$PYTHON" "$SCRIPTS/diag_summarize.py" "$DIAG_ROOT"

echo ""
echo "================================================================"
echo "Diagnostics complete. Output: $DIAG_ROOT"
echo "Summary:                       $DIAG_ROOT/SUMMARY.md"
echo "================================================================"
