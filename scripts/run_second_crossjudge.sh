#!/bin/bash
# Run a second cross-judge against existing artifacts from the overnight run.
#
# Triangulates the GPT-5 cross-judge result already in
#   logs/seed2_constantlr_total_absolute_20260504-071351/SUMMARY.md
# by re-judging the same val examples with a different model. Pinference
# (api.pinference.ai) routes to multiple vendors via one OpenAI-compatible
# endpoint, so swapping the judge is a CLI-only change — no edits to
# scripts/cross_judge_rubric.py needed.
#
# Usage:
#   PRIME_API_KEY=... ./scripts/run_second_crossjudge.sh
#       [--judge-model <id>] [--label <existing-overnight-label>]
#       [--phase5-jsonl <dir>] [--limit-prior <N>] [--limit-new <N>]
#
# Default: judge=anthropic/claude-sonnet-4.6, re-judges the original Phase 1
# wandb subset (n=80) and the full Phase 2 dump (n=819). If --phase5-jsonl
# points at a directory of validation .jsonl files, it also re-judges that.
#
# Outputs land beside the existing GPT-5 results so a single `ls` shows both
# judges' summaries side-by-side. SUMMARY.md is intentionally NOT regenerated;
# the comparison table is composed by hand from the resulting JSONs.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/root/rubric-policy-optimization}"
SDPO_ROOT="$REPO_ROOT/SDPO"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# Defaults — override with CLI flags below.
JUDGE_MODEL="anthropic/claude-sonnet-4.6"
JUDGE_BASE_URL="https://api.pinference.ai/api/v1"
JUDGE_API_KEY_ENV="PRIME_API_KEY"
LABEL="seed2_constantlr_total_absolute_20260504-071351"
PRIOR_RUN_PATH="alexander-mader-yale-university/SDPO-RUPO-litbench/5uwbvhqf"
PHASE5_JSONL=""
# n=80 matches the GPT-5 cross-judge — the wandb table only has 80 anyway.
LIMIT_PRIOR=80
# Bump from the orchestrator's default 200 to the full val set for tighter CIs.
LIMIT_NEW=819
MAX_CONCURRENT=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --judge-model)        JUDGE_MODEL="$2"; shift 2 ;;
        --judge-base-url)     JUDGE_BASE_URL="$2"; shift 2 ;;
        --judge-api-key-env)  JUDGE_API_KEY_ENV="$2"; shift 2 ;;
        --label)              LABEL="$2"; shift 2 ;;
        --phase5-jsonl)       PHASE5_JSONL="$2"; shift 2 ;;
        --limit-prior)        LIMIT_PRIOR="$2"; shift 2 ;;
        --limit-new)          LIMIT_NEW="$2"; shift 2 ;;
        --max-concurrent)     MAX_CONCURRENT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

LOG_DIR="$SDPO_ROOT/logs/$LABEL"
PARQUET_TEST="$SDPO_ROOT/datasets/dpo_to_rupo_litbench_ha_with_baseline_criteria_total_absolute/test.parquet"

if [[ ! -d "$LOG_DIR" ]]; then
    echo "ERROR: Log dir does not exist: $LOG_DIR" >&2
    echo "Pass --label <existing-orchestrator-label> if running against a different overnight." >&2
    exit 1
fi
if [[ -z "${!JUDGE_API_KEY_ENV:-}" ]]; then
    echo "ERROR: \$$JUDGE_API_KEY_ENV is not set." >&2
    exit 1
fi
if [[ ! -f "$PARQUET_TEST" ]]; then
    echo "ERROR: Missing test parquet at $PARQUET_TEST." >&2
    exit 1
fi

# Filesystem-safe model tag for the output filename. "anthropic/claude-sonnet-4.6"
# becomes "anthropic_claude-sonnet-4_6" so a directory listing groups results
# by judge unambiguously.
MODEL_TAG="$(echo "$JUDGE_MODEL" | tr '/.' '__')"

echo "============================================================"
echo "Second cross-judge run"
echo "  judge_model:   $JUDGE_MODEL"
echo "  judge_base:    $JUDGE_BASE_URL"
echo "  api_key_env:   $JUDGE_API_KEY_ENV"
echo "  log_dir:       $LOG_DIR"
echo "  prior_limit:   $LIMIT_PRIOR (against $PRIOR_RUN_PATH)"
echo "  new_limit:     $LIMIT_NEW (against $LOG_DIR/val_generations)"
if [[ -n "$PHASE5_JSONL" ]]; then
    echo "  phase5_jsonl:  $PHASE5_JSONL"
fi
echo "  model_tag:     $MODEL_TAG"
echo "============================================================"

# Run a single cross-judge invocation.
#   $1 = friendly name (for log path)
#   $2 = output JSON path
#   $3.. = additional args appended verbatim (--source, --jsonl/--wandb-run, --limit, etc.)
run_one() {
    local name="$1"; shift
    local out="$1"; shift
    local log="$LOG_DIR/crossjudge_${name}_${MODEL_TAG}.log"
    echo ""
    echo "[$name] $(date '+%H:%M:%S')  → $out"
    "$SDPO_ROOT/.venv/bin/python" "$SDPO_ROOT/scripts/cross_judge_rubric.py" \
        --judge-model "$JUDGE_MODEL" \
        --judge-base-url "$JUDGE_BASE_URL" \
        --judge-api-key-env "$JUDGE_API_KEY_ENV" \
        --max-concurrent "$MAX_CONCURRENT" \
        --output "$out" \
        "$@" \
        > "$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[$name] OK  log=$log"
    else
        echo "[$name] FAILED rc=$rc  log=$log"
    fi
    return $rc
}

PRIOR_OUT="$LOG_DIR/crossjudge_prior_5uwbvhqf_${MODEL_TAG}.json"
NEW_OUT="$LOG_DIR/crossjudge_new_${LABEL}_${MODEL_TAG}.json"

run_one "prior" "$PRIOR_OUT" \
    --source wandb \
    --wandb-run "$PRIOR_RUN_PATH" \
    --parquet "$PARQUET_TEST" \
    --limit "$LIMIT_PRIOR"
PRIOR_RC=$?

run_one "new" "$NEW_OUT" \
    --source jsonl \
    --jsonl "$LOG_DIR/val_generations" \
    --jsonl-step latest \
    --limit "$LIMIT_NEW"
NEW_RC=$?

if [[ -n "$PHASE5_JSONL" ]]; then
    if [[ -d "$PHASE5_JSONL" ]] && compgen -G "$PHASE5_JSONL/*.jsonl" > /dev/null; then
        PHASE5_OUT="$LOG_DIR/crossjudge_phase5_${MODEL_TAG}.json"
        run_one "phase5" "$PHASE5_OUT" \
            --source jsonl \
            --jsonl "$PHASE5_JSONL" \
            --jsonl-step latest \
            --limit "$LIMIT_NEW"
        PHASE5_RC=$?
    else
        echo "[phase5] SKIPPED — no .jsonl files in $PHASE5_JSONL"
        PHASE5_RC=255
    fi
else
    PHASE5_RC=0  # not requested
fi

# Compact summary so the user can scan results without re-opening every JSON.
echo ""
echo "============================================================"
echo "## Summary ($JUDGE_MODEL)"
"$SDPO_ROOT/.venv/bin/python" - "$PRIOR_OUT" "$NEW_OUT" "${PHASE5_OUT:-}" <<'PY'
import json, sys
labels = ["prior 5uwbvhqf", "new (Phase 2)", "Phase 5 rerun"]
print(f"{'config':24s} {'n_eval':>8s} {'self-judge':>12s} {'cross-judge':>14s}")
print("-" * 62)
for path, label in zip(sys.argv[1:], labels):
    if not path:
        continue
    try:
        d = json.load(open(path))["summary"]
    except Exception as e:
        print(f"{label:24s} {'ERR':>8s}  {str(e)[:40]}")
        continue
    n = d.get("n_evaluated", 0)
    cross = d.get("mean_crossjudge_reward")
    orig = d.get("mean_original_reward")
    cs = f"{cross:.4f}" if isinstance(cross, (int, float)) and cross == cross else "N/A"
    os_ = f"{orig:.4f}" if isinstance(orig, (int, float)) and orig == orig else "N/A"
    print(f"{label:24s} {n:>8d} {os_:>12s} {cs:>14s}")
PY

echo ""
echo "Return codes: prior=$PRIOR_RC  new=$NEW_RC  phase5=$PHASE5_RC"
echo "Finished: $(date)"

if [[ $PRIOR_RC -ne 0 || $NEW_RC -ne 0 || $PHASE5_RC -ne 0 ]]; then
    exit 1
fi
exit 0
