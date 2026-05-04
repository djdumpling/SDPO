#!/bin/bash
# Overnight rigor battery for the SDPO RuPO submission.
#
# Six phases, sequenced so a single GPU box can run all of it overnight:
#   1. (BG)    Cross-judge prior winning run 5uwbvhqf with GPT-5. Validates
#              the API key while training runs in the foreground.
#   2. (FG)    Train seed=2 (constant LR + criteria_total_absolute) and dump
#              the full 819-example val outputs to JSONL so Phase 3 can
#              re-judge them offline.
#   3. (FG)    Cross-judge the seed=2 run with GPT-5.
#   4. (FG)    Train seed=3 of the same config — gives n=3 across
#              {5uwbvhqf seed=1, Phase 2 seed=2, Phase 4 seed=3} for the
#              headline number with proper variance.
#   5. (FG)    Train criteria_margin + constant LR — disconfounds the original
#              5-shape ablation (which mixed cosine LR with reward shape) so
#              we can claim total_absolute wins on its own merit.
#   6. (final) Pull self-judge + cross-judge numbers into SUMMARY.md.
#
# No model checkpoints are saved (SAVE_FREQ=-1 is the launcher default), per
# user preference. Phases 4 and 5 also turn off rollout/validation dumps since
# their only artifact is the wandb val/reward number.
#
# Outputs land under SDPO/logs/<LABEL>/. The wandb experiment_name and rollout
# dump folder for each training phase carry distinct labels so they don't
# collide with prior runs or with each other.
#
# Required env: PRIME_API_KEY (Prime Intellect inference API key).
# Optional env:
#   SEED                          (default: 2; only affects Phase 2)
#   CROSSJUDGE_NEW_LIMIT          (default: 200; cap on Phase 3 examples)
#   CROSSJUDGE_PRIOR_LIMIT        (default: 80; the wandb table only has 80)
#   CROSSJUDGE_MAX_CONCURRENT     (default: 32)
#   CROSSJUDGE_PRIOR_TIMEOUT      (default: 90m, GPT-5 API only)
#   CROSSJUDGE_NEW_TIMEOUT        (default: 180m, GPT-5 API only)
#
# Training phases have NO timeout — let them run as long as needed; if any
# fail, the orchestrator logs the RC and proceeds to the next phase.

set -uo pipefail

# -----------------------------------------------------------------------------
# Setup and sanity
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-/root/rubric-policy-optimization}"
SDPO_ROOT="$REPO_ROOT/SDPO"

# Source the repo .env so PRIME_API_KEY, WANDB_API_KEY, etc. are visible to
# both the orchestrator and the cross-judge subprocesses. The slurm wrapper
# also sources it later for training, but Phase 1 runs before that.
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
SEED_OVERRIDE=${SEED:-2}
CROSSJUDGE_NEW_LIMIT=${CROSSJUDGE_NEW_LIMIT:-200}
CROSSJUDGE_PRIOR_LIMIT=${CROSSJUDGE_PRIOR_LIMIT:-80}
CROSSJUDGE_MAX_CONCURRENT=${CROSSJUDGE_MAX_CONCURRENT:-32}
# Cross-judge timeouts. Wall-clock cap = call_timeout * ceil(N_calls/concurrency)
# in the worst case. We pick generous per-phase caps so a slow Prime endpoint
# can't leave the whole orchestrator hung past sunrise. Training phases
# intentionally have NO timeout — let each run as long as it needs; if one
# fails the orchestrator continues to the next.
CROSSJUDGE_PRIOR_TIMEOUT=${CROSSJUDGE_PRIOR_TIMEOUT:-90m}
CROSSJUDGE_NEW_TIMEOUT=${CROSSJUDGE_NEW_TIMEOUT:-180m}
PRIOR_RUN_PATH="alexander-mader-yale-university/SDPO-RUPO-litbench/5uwbvhqf"
PARQUET_TEST="$SDPO_ROOT/datasets/dpo_to_rupo_litbench_ha_with_baseline_criteria_total_absolute/test.parquet"

# Fail fast on missing prerequisites — better than hitting them at 4am.
if [[ -z "${PRIME_API_KEY:-}" ]]; then
    echo "ERROR: PRIME_API_KEY is not set. Cross-judge with GPT-5 needs it." >&2
    exit 1
fi
if [[ ! -f "$PARQUET_TEST" ]]; then
    echo "ERROR: Missing $PARQUET_TEST. Run prepare_litbench_ha_with_baseline.sh first." >&2
    exit 1
fi
if [[ ! -f "$SDPO_ROOT/scripts/cross_judge_rubric.py" ]]; then
    echo "ERROR: Missing $SDPO_ROOT/scripts/cross_judge_rubric.py" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Descriptive label so wandb run name and dump folders don't collide with prior
# runs. Embedded in EXP_SUFFIX (consumed by the launcher) and in folder paths.
# -----------------------------------------------------------------------------

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LABEL="seed${SEED_OVERRIDE}_constantlr_total_absolute_${TIMESTAMP}"
LOG_DIR="$SDPO_ROOT/logs/$LABEL"
ROLLOUT_DUMP_DIR="/logs/${LABEL}_rollout_dump"
VALIDATION_DUMP_DIR="$LOG_DIR/val_generations"
SUMMARY_FILE="$LOG_DIR/SUMMARY.md"
mkdir -p "$LOG_DIR"
mkdir -p "$VALIDATION_DUMP_DIR"

# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------

note() { echo "$@" | tee -a "$SUMMARY_FILE"; }

# wait_for_judge_port_free: defensive cleanup between training phases.
# If a previous phase timed out, the orchestrator's `timeout` only signals the
# slurm wrapper, not the launcher subprocess underneath. The wrapper's EXIT
# trap kills its own judge, but a hung Ray worker can keep port 8000 bound.
# We wait up to 60s for the port to free naturally, then SIGKILL any vLLM
# server still owned by THIS user (so we never touch other tenants on the
# shared box). Safe to call when nothing is running.
wait_for_judge_port_free() {
    for _ in $(seq 1 30); do
        if ! ss -ltn 2>/dev/null | grep -q ':8000 '; then return 0; fi
        sleep 2
    done
    pkill -KILL -U "$(whoami)" -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -KILL -U "$(whoami)" -f "ray::TaskRunner" 2>/dev/null || true
    pkill -KILL -U "$(whoami)" -f "ray::AgentLoopWorker" 2>/dev/null || true
    sleep 5
}

note "=============================================================="
note "Overnight orchestrator: $LABEL"
note "Started:    $(date)"
note "Logs dir:   $LOG_DIR"
note "Rollouts:   $ROLLOUT_DUMP_DIR"
note "Val dumps:  $VALIDATION_DUMP_DIR"
note "Seed:       $SEED_OVERRIDE"
note "Prior run:  $PRIOR_RUN_PATH"
note "Judge:      openai/gpt-5 via https://api.pinference.ai/api/v1"
note "=============================================================="

# -----------------------------------------------------------------------------
# Phase 1 (background): cross-judge prior winning run with GPT-5.
# Validates the API key + model availability while training runs. Whatever
# happens here, training in Phase 2 still proceeds — we don't want a transient
# API issue to cost us 3 hours of GPU time.
# -----------------------------------------------------------------------------

PHASE1_LOG="$LOG_DIR/phase1_crossjudge_prior.log"
PHASE1_OUT="$LOG_DIR/crossjudge_prior_5uwbvhqf_gpt5.json"

note ""
note "[Phase 1] BG: cross-judge prior run $PRIOR_RUN_PATH ($(date '+%H:%M:%S'))"
(
    timeout --signal=TERM "$CROSSJUDGE_PRIOR_TIMEOUT" \
    "$SDPO_ROOT/.venv/bin/python" "$SDPO_ROOT/scripts/cross_judge_rubric.py" \
        --source wandb \
        --wandb-run "$PRIOR_RUN_PATH" \
        --parquet "$PARQUET_TEST" \
        --judge-model "openai/gpt-5" \
        --judge-base-url "https://api.pinference.ai/api/v1" \
        --judge-api-key-env "PRIME_API_KEY" \
        --max-concurrent "$CROSSJUDGE_MAX_CONCURRENT" \
        --limit "$CROSSJUDGE_PRIOR_LIMIT" \
        --output "$PHASE1_OUT" \
        > "$PHASE1_LOG" 2>&1
    echo $? > "$LOG_DIR/.phase1_rc"
) &
PHASE1_PID=$!
note "    pid=$PHASE1_PID  log=$PHASE1_LOG  out=$PHASE1_OUT"

# -----------------------------------------------------------------------------
# Phase 2 (foreground): training. Pass our descriptive label as EXP_SUFFIX so
# the wandb run name reflects this experiment, and dump validation generations
# to disk so Phase 3 can re-judge them offline. timeout guards against hangs.
# -----------------------------------------------------------------------------

PHASE2_LOG="$LOG_DIR/phase2_training.log"
note ""
note "[Phase 2] FG: training seed=$SEED_OVERRIDE ($(date '+%H:%M:%S'))"
wait_for_judge_port_free

cd "$SDPO_ROOT"
SEED="$SEED_OVERRIDE" \
EXP_SUFFIX="$LABEL" \
ROLLOUT_DATA_DIR="$ROLLOUT_DUMP_DIR" \
VALIDATION_DATA_DIR="$VALIDATION_DUMP_DIR" \
    bash train_qwen35_9b_litbench_8xh200_external_judge.slurm \
    > "$PHASE2_LOG" 2>&1
PHASE2_RC=$?

if [[ $PHASE2_RC -eq 0 ]]; then
    note "[Phase 2] training finished cleanly. log=$PHASE2_LOG"
else
    note "[Phase 2] WARN: training exited rc=$PHASE2_RC. log=$PHASE2_LOG"
fi

# -----------------------------------------------------------------------------
# Phase 3 (foreground): cross-judge the new run's final-step val JSONL.
# We let trainer.validation_data_dir do the hard work of writing every val
# example to disk — much cleaner than reconstructing from wandb tables.
# -----------------------------------------------------------------------------

note ""
note "[Phase 3] cross-judge new run val generations ($(date '+%H:%M:%S'))"
PHASE3_LOG="$LOG_DIR/phase3_crossjudge_new.log"
PHASE3_OUT="$LOG_DIR/crossjudge_new_${LABEL}_gpt5.json"

if compgen -G "$VALIDATION_DUMP_DIR/*.jsonl" > /dev/null; then
    timeout --signal=TERM "$CROSSJUDGE_NEW_TIMEOUT" \
    "$SDPO_ROOT/.venv/bin/python" "$SDPO_ROOT/scripts/cross_judge_rubric.py" \
        --source jsonl \
        --jsonl "$VALIDATION_DUMP_DIR" \
        --jsonl-step latest \
        --judge-model "openai/gpt-5" \
        --judge-base-url "https://api.pinference.ai/api/v1" \
        --judge-api-key-env "PRIME_API_KEY" \
        --max-concurrent "$CROSSJUDGE_MAX_CONCURRENT" \
        --limit "$CROSSJUDGE_NEW_LIMIT" \
        --output "$PHASE3_OUT" \
        > "$PHASE3_LOG" 2>&1
    PHASE3_RC=$?
    if [[ $PHASE3_RC -eq 0 ]]; then
        note "[Phase 3] OK. out=$PHASE3_OUT"
    else
        note "[Phase 3] FAILED rc=$PHASE3_RC. log=$PHASE3_LOG"
    fi
else
    note "[Phase 3] SKIPPED — no .jsonl files in $VALIDATION_DUMP_DIR (training likely failed early)"
    PHASE3_RC=255
fi

# -----------------------------------------------------------------------------
# Phase 4 (foreground): seed=3 training. Combined with seed=1 (5uwbvhqf) and
# seed=2 (Phase 2), this gives n=3 for the headline number — the minimum to
# claim variance honestly. Same config as Phase 2 except for the seed; rollout
# and validation dumps are disabled because we only need the wandb val/reward
# number from this run.
# -----------------------------------------------------------------------------

PHASE4_SUFFIX="seed3_constantlr_total_absolute_${TIMESTAMP}"
PHASE4_LOG="$LOG_DIR/phase4_training_seed3.log"
note ""
note "[Phase 4] FG: training seed=3 (constant LR + total_absolute) ($(date '+%H:%M:%S'))"
note "    EXP_SUFFIX=$PHASE4_SUFFIX"
wait_for_judge_port_free

cd "$SDPO_ROOT"
SEED=3 \
EXP_SUFFIX="$PHASE4_SUFFIX" \
JUDGE_REWARD_TYPE_TRAIN=criteria_total_absolute \
JUDGE_REWARD_TYPE_VAL=criteria_total_absolute \
LR_SCHEDULER_TYPE=constant \
ROLLOUT_DATA_DIR=null \
VALIDATION_DATA_DIR=null \
    bash train_qwen35_9b_litbench_8xh200_external_judge.slurm \
    > "$PHASE4_LOG" 2>&1
PHASE4_RC=$?
if [[ $PHASE4_RC -eq 0 ]]; then
    note "[Phase 4] training finished cleanly. log=$PHASE4_LOG"
else
    note "[Phase 4] WARN: training exited rc=$PHASE4_RC. log=$PHASE4_LOG"
fi

# -----------------------------------------------------------------------------
# Phase 5 (foreground): criteria_margin reward + constant LR. The original
# 5-shape ablation was confounded by cosine LR (it was the late-training
# degrader); rerunning the criteria_margin baseline at constant LR shows
# whether total_absolute still dominates once the schedule is held fixed.
# Different parquet ($JUDGE_REWARD_TYPE_TRAIN=criteria_margin) is required.
# -----------------------------------------------------------------------------

PHASE5_SUFFIX="criteriamargin_constantlr_${TIMESTAMP}"
PHASE5_LOG="$LOG_DIR/phase5_training_criteriamargin.log"
PARQUET_CM_TRAIN="$SDPO_ROOT/datasets/dpo_to_rupo_litbench_ha_with_baseline_criteria_margin/train.parquet"
PARQUET_CM_TEST="$SDPO_ROOT/datasets/dpo_to_rupo_litbench_ha_with_baseline_criteria_margin/test.parquet"
note ""
note "[Phase 5] FG: training criteria_margin + constant LR ($(date '+%H:%M:%S'))"
note "    EXP_SUFFIX=$PHASE5_SUFFIX"
wait_for_judge_port_free

if [[ ! -f "$PARQUET_CM_TRAIN" || ! -f "$PARQUET_CM_TEST" ]]; then
    note "[Phase 5] SKIPPED — missing criteria_margin parquet at $(dirname "$PARQUET_CM_TRAIN")"
    PHASE5_RC=255
else
    cd "$SDPO_ROOT"
    SEED=1 \
    EXP_SUFFIX="$PHASE5_SUFFIX" \
    JUDGE_REWARD_TYPE_TRAIN=criteria_margin \
    JUDGE_REWARD_TYPE_VAL=criteria_margin \
    LR_SCHEDULER_TYPE=constant \
    ROLLOUT_DATA_DIR=null \
    VALIDATION_DATA_DIR=null \
        bash train_qwen35_9b_litbench_8xh200_external_judge.slurm \
        > "$PHASE5_LOG" 2>&1
    PHASE5_RC=$?
    if [[ $PHASE5_RC -eq 0 ]]; then
        note "[Phase 5] training finished cleanly. log=$PHASE5_LOG"
    else
        note "[Phase 5] WARN: training exited rc=$PHASE5_RC. log=$PHASE5_LOG"
    fi
fi

# -----------------------------------------------------------------------------
# Phase 6: gather Phase 1 result, pull self-judge val/reward for every training
# phase from wandb, write final SUMMARY.md.
# -----------------------------------------------------------------------------

note ""
note "[Phase 6] waiting for Phase 1 to finish..."
wait "$PHASE1_PID" 2>/dev/null
PHASE1_RC=$(cat "$LOG_DIR/.phase1_rc" 2>/dev/null || echo "?")
note "[Phase 1] rc=$PHASE1_RC  log=$PHASE1_LOG"

# Aggregate results: cross-judge JSONs (Phases 1 + 3) + per-training wandb
# val/reward at the final step (Phases 2, 4, 5). The wandb pull tolerates
# missing runs so the orchestrator never fails just because a single phase
# was skipped or hasn't synced yet.
note ""
note "## Cross-judge results (preference accuracy on val examples)"
"$SDPO_ROOT/.venv/bin/python" - "$PHASE1_OUT" "$PHASE3_OUT" <<'PY' 2>/dev/null | tee -a "$SUMMARY_FILE"
import json, sys
labels = ["prior 5uwbvhqf (cosine LR, seed=1)", "Phase 2 seed=2 run (constant LR)"]
print(f"{'config':52s} {'n_eval':>8s} {'self-judge':>12s} {'gpt-5 cross-judge':>20s}")
print("-" * 94)
for path, label in zip(sys.argv[1:], labels):
    try:
        d = json.load(open(path))["summary"]
        n = d.get("n_evaluated", 0)
        cross = d.get("mean_crossjudge_reward")
        orig = d.get("mean_original_reward")
        cross_s = f"{cross:.4f}" if isinstance(cross, (int, float)) and cross == cross else "N/A"
        orig_s = f"{orig:.4f}" if isinstance(orig, (int, float)) and orig == orig else "N/A"
        print(f"{label:52s} {n:>8d} {orig_s:>12s} {cross_s:>20s}")
    except Exception as e:
        print(f"{label:52s} {'ERR':>8s} {str(e)[:40]:>12s}")
PY

note ""
note "## Self-judge val/reward per training run (final step from wandb)"
"$SDPO_ROOT/.venv/bin/python" - "$LABEL" "$PHASE4_SUFFIX" "$PHASE5_SUFFIX" <<'PY' 2>/dev/null | tee -a "$SUMMARY_FILE"
import os, sys
# Source the .env for WANDB_API_KEY since this is a fresh subprocess.
for line in open('/root/rubric-policy-optimization/.env'):
    if '=' in line and not line.strip().startswith('#'):
        k, v = line.strip().split('=', 1)
        os.environ.setdefault(k, v.strip("'\""))

phase2_suffix, phase4_suffix, phase5_suffix = sys.argv[1], sys.argv[2], sys.argv[3]
EXP_PATTERN = "SDPO-RUPO-bs12-n4-alpha0.5-lr2e-6-lora0-Qwen-Qwen3-8B-{suf}"

configs = [
    ("prior 5uwbvhqf (constant LR, seed=1)",   "alexander-mader-yale-university/SDPO-RUPO-litbench/5uwbvhqf"),
    ("Phase 2 (constant LR, seed=2)",          EXP_PATTERN.format(suf=phase2_suffix)),
    ("Phase 4 (constant LR, seed=3)",          EXP_PATTERN.format(suf=phase4_suffix)),
    ("Phase 5 (criteria_margin, constant LR)", EXP_PATTERN.format(suf=phase5_suffix)),
]
try:
    import wandb
    api = wandb.Api()
except Exception as e:
    print(f"  wandb unavailable: {e}")
    sys.exit(0)

def find_run(handle):
    """Resolve a wandb handle. If full entity/project/id, fetch directly; else
    treat as display_name and pick the MOST RECENTLY created run with that
    name (orchestrator suffixes carry a timestamp so collisions don't happen
    in practice, but ordering by created_at avoids picking a stale legacy run
    if a name ever did repeat)."""
    if "/" in handle and handle.count("/") == 2:
        return api.run(handle)
    runs = api.runs("alexander-mader-yale-university/SDPO-RUPO-litbench",
                     filters={"display_name": handle}, per_page=10, order="-created_at")
    runs = list(runs)
    return runs[0] if runs else None

print(f"{'config':52s} {'val/reward':>12s} {'final step':>12s} {'wandb id':>14s}")
print("-" * 94)
for label, handle in configs:
    try:
        run = find_run(handle)
        if run is None:
            print(f"{label:52s} {'N/A':>12s} {'N/A':>12s} {'<no run>':>14s}")
            continue
        h = run.history(keys=["_step", "val-core/dpo_to_rupo/reward/mean@1"], pandas=False, samples=2000)
        rows = [r for r in h if r.get("val-core/dpo_to_rupo/reward/mean@1") is not None]
        if not rows:
            print(f"{label:52s} {'(no val)':>12s} {'-':>12s} {run.id:>14s}")
            continue
        last = max(rows, key=lambda r: r.get("_step", 0))
        v = last['val-core/dpo_to_rupo/reward/mean@1']
        s = last.get("_step", "?")
        print(f"{label:52s} {v:>12.4f} {s:>12} {run.id:>14s}")
    except Exception as e:
        print(f"{label:52s} {'ERR':>12s} {str(e)[:24]:>12s}")
PY

note ""
note "## Files"
note "- log dir:                $LOG_DIR"
note "- Phase 1 cross-judge:    $PHASE1_OUT  (log: $PHASE1_LOG)"
note "- Phase 2 training log:   $PHASE2_LOG"
note "- Phase 2 val JSONLs:     $VALIDATION_DUMP_DIR"
note "- Phase 3 cross-judge:    $PHASE3_OUT  (log: $PHASE3_LOG)"
note "- Phase 4 training log:   $PHASE4_LOG"
note "- Phase 5 training log:   $PHASE5_LOG"
note ""
note "Phase return codes: P1=$PHASE1_RC P2=$PHASE2_RC P3=$PHASE3_RC P4=$PHASE4_RC P5=$PHASE5_RC"
note "Finished: $(date)"

# Non-zero overall exit if anything broke, but only after SUMMARY.md is fully
# written so the user can see partial results.
if [[ "$PHASE1_RC" != "0" ]] || [[ "$PHASE2_RC" -ne 0 ]] || [[ "$PHASE3_RC" -ne 0 ]] \
        || [[ "$PHASE4_RC" -ne 0 ]] || [[ "$PHASE5_RC" -ne 0 ]]; then
    exit 1
fi
exit 0
