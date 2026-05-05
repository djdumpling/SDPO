"""Phase 4 helper — aggregate k cross_judge_rubric.py runs into one ensembled result.

Reads K JSON files (one per judge resample) and produces a single JSON whose
per-row scores are the mean across the K resamples. Re-derives the criteria-
total-absolute reward from averaged criterion scores and reports headline:

  - per-row accuracy under each individual k (test-retest at row level)
  - per-row accuracy under k-averaged scores (the OpenRubrics-style fix)
  - ensembling lift = avg(individual) → ensembled
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def absolute_reward(gap: float) -> float:
    if gap > 0: return 1.0
    if gap == 0: return 0.5
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="K cross_judge_rubric output JSONs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = []
    for p in args.inputs:
        d = json.loads(Path(p).read_text())
        runs.append(d)
    K = len(runs)
    print(f"  loaded {K} resample runs")

    # Each row in each run has fields including chosen_score, rejected_score, accuracy.
    # Index by row id (use jsonl_idx or fall back to row order)
    by_id = defaultdict(list)  # id -> list of K rows
    for run_idx, run in enumerate(runs):
        for i, row in enumerate(run.get("rows", [])):
            rid = row.get("jsonl_idx", row.get("idx", i))
            by_id[rid].append(row)

    rows_out = []
    individual_acc = [[] for _ in range(K)]
    ensemble_acc = []
    individual_margins = [[] for _ in range(K)]
    ensemble_margins = []
    for rid, group in by_id.items():
        if len(group) < K:
            continue  # keep only rows present in ALL runs
        # Average chosen_score and rejected_score across K resamples
        cs = [g.get("chosen_score") for g in group if g.get("chosen_score") is not None]
        rs = [g.get("rejected_score") for g in group if g.get("rejected_score") is not None]
        if not cs or not rs:
            continue
        cs_ens = statistics.fmean(cs)
        rs_ens = statistics.fmean(rs)
        margin_ens = cs_ens - rs_ens
        acc_ens = absolute_reward(margin_ens)

        for k_i, g in enumerate(group):
            ch_k, rj_k = g.get("chosen_score"), g.get("rejected_score")
            if ch_k is not None and rj_k is not None:
                individual_acc[k_i].append(absolute_reward(ch_k - rj_k))
                individual_margins[k_i].append(ch_k - rj_k)

        ensemble_acc.append(acc_ens)
        ensemble_margins.append(margin_ens)
        rows_out.append({
            "id": rid,
            "chosen_scores_k": cs, "rejected_scores_k": rs,
            "chosen_score_ensemble": cs_ens,
            "rejected_score_ensemble": rs_ens,
            "margin_ensemble": margin_ens,
            "accuracy_ensemble": acc_ens,
        })

    individual_means = [statistics.fmean(a) if a else None for a in individual_acc]
    ensemble_mean = statistics.fmean(ensemble_acc) if ensemble_acc else None
    individual_avg = statistics.fmean(
        [m for m in individual_means if m is not None]) if individual_means else None

    headline = {
        "K": K,
        "n_rows_kept": len(rows_out),
        "individual_run_accuracy": individual_means,
        "individual_avg_accuracy": individual_avg,
        "ensemble_accuracy": ensemble_mean,
        "ensembling_lift_pp": (
            (ensemble_mean - individual_avg) * 100
            if (ensemble_mean is not None and individual_avg is not None) else None),
        "verdict": None,
    }
    lift = headline["ensembling_lift_pp"] or 0
    if lift > 1.0:
        headline["verdict"] = (
            f"H1 supported — ensembling k={K} lifts accuracy by {lift:.1f}pp over single-call. "
            f"Implementing judge ensembling in training would reduce reward noise meaningfully.")
    elif lift > 0:
        headline["verdict"] = f"H1 marginal — k={K} lift = {lift:.1f}pp. Ensembling helps a little; not the dominant fix."
    else:
        headline["verdict"] = f"H1 NOT supported by ensembling alone — lift = {lift:.1f}pp. Judge noise may not be the main issue."

    out = {"headline": headline, "rows": rows_out}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
