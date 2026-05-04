"""Apples-to-apples 5uwbvhqf vs Phase 2 cross-judge comparison.

Aggregates the two-judge cross-judge results on the same 20 prompts at the
same training steps (25 and 50, the only steps captured for 5uwbvhqf), and
reports paired bootstrap CIs.

Aggregation rule. 5uwbvhqf has 4 rollouts per prompt (3 at step 25 + 1 at
step 50). Phase 2 has 2 rollouts per prompt (1 at step 25 + 1 at step 50).
We average within (run, prompt, step) cell first, then either:
  - per-step comparison: step 25 alone (3-vs-1 rollouts averaged per prompt)
  - per-step comparison: step 50 alone (1-vs-1)
  - pooled: simple average of step 25 and step 50 means per prompt

Usage:
  python scripts/compare_matched_crossjudge.py --judge gpt5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

LOG_DIR = "/root/rubric-policy-optimization/SDPO/logs/seed2_constantlr_total_absolute_20260504-071351"

JUDGE_FILES = {
    "gpt5": (
        f"{LOG_DIR}/crossjudge_prior_5uwbvhqf_gpt5.json",
        f"{LOG_DIR}/crossjudge_matched20_gpt5.json",
    ),
    "sonnet": (
        f"{LOG_DIR}/crossjudge_prior_5uwbvhqf_anthropic_claude-sonnet-4_6.json",
        f"{LOG_DIR}/crossjudge_matched20_anthropic_claude-sonnet-4_6.json",
    ),
}

SCAFFOLD_MARKERS = ["\n\nHere is one candidate response", "\n\nHere is the prompt", "\n<candidate_a>"]


def normalize(p: str) -> str:
    s = p.strip()
    for m in SCAFFOLD_MARKERS:
        i = s.find(m)
        if i != -1:
            s = s[:i]
    return s.strip()


def collect(json_path: str, normalize_prompt: bool):
    """Returns dict[(prompt, step)] -> list[reward]."""
    d = json.load(open(json_path))
    by_cell = defaultdict(list)
    for r in d["rows"]:
        if not r.get("judge_called") or r.get("judge_error") or "reward" not in r:
            continue
        p = r["prompt"] or ""
        if normalize_prompt:
            p = normalize(p)
        else:
            p = p.strip()
        step = r.get("step")
        by_cell[(p, step)].append(float(r["reward"]))
    return by_cell


def paired_bootstrap_ci(deltas: list[float], n_boot: int = 5000, alpha: float = 0.05):
    n = len(deltas)
    rng = random.Random(0)
    samples = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    samples.sort()
    return statistics.fmean(deltas), samples[int(n_boot * alpha / 2)], samples[int(n_boot * (1 - alpha / 2))]


def report_one_judge(judge_label: str, prior_path: str, matched_path: str, n_boot: int = 5000):
    print(f"\n========================================")
    print(f"Judge: {judge_label}")
    print(f"  prior:   {prior_path.split('/')[-1]}")
    print(f"  matched: {matched_path.split('/')[-1]}")
    print(f"========================================")

    prior_cells = collect(prior_path, normalize_prompt=True)   # prior prompts have scaffold
    matched_cells = collect(matched_path, normalize_prompt=False)  # JSONL prompts already clean

    prior_prompts = sorted({p for (p, s) in prior_cells.keys()})
    matched_prompts = sorted({p for (p, s) in matched_cells.keys()})
    common = sorted(set(prior_prompts) & set(matched_prompts))
    print(f"  prior unique prompts:   {len(prior_prompts)}")
    print(f"  matched unique prompts: {len(matched_prompts)}")
    print(f"  common prompts:         {len(common)}")

    # Per-cell means.
    def cell_mean(cells, prompt, step):
        if (prompt, step) not in cells: return None
        v = cells[(prompt, step)]
        return sum(v) / len(v)

    print(f"\n  Per-prompt cell-mean comparison:")
    print(f"    {'step':>5s}  {'n_prompts':>10s}  {'5uwbvhqf':>12s}  {'Phase 2':>12s}  {'delta (5u−Ph2)':>18s}  {'95% CI':>20s}")
    for step in [25, 50]:
        deltas = []
        priors = []
        phs = []
        for p in common:
            pv = cell_mean(prior_cells, p, step)
            mv = cell_mean(matched_cells, p, step)
            if pv is None or mv is None: continue
            priors.append(pv); phs.append(mv); deltas.append(pv - mv)
        if not deltas:
            print(f"    {step:>5d}  no overlap")
            continue
        prior_m = statistics.fmean(priors)
        ph_m = statistics.fmean(phs)
        d_m, lo, hi = paired_bootstrap_ci(deltas, n_boot=n_boot)
        print(f"    {step:>5d}  {len(deltas):>10d}  {prior_m:>12.4f}  {ph_m:>12.4f}  {d_m:>+18.4f}  [{lo:>+.4f},{hi:>+.4f}]")

    # Pooled across-step (per-prompt average of step 25 and step 50 means).
    print(f"\n  Per-prompt pooled (step 25 mean + step 50 mean averaged):")
    deltas = []
    priors = []
    phs = []
    for p in common:
        p25 = cell_mean(prior_cells, p, 25); p50 = cell_mean(prior_cells, p, 50)
        m25 = cell_mean(matched_cells, p, 25); m50 = cell_mean(matched_cells, p, 50)
        if None in (p25, p50, m25, m50): continue
        pv = (p25 + p50) / 2
        mv = (m25 + m50) / 2
        priors.append(pv); phs.append(mv); deltas.append(pv - mv)
    if deltas:
        prior_m = statistics.fmean(priors)
        ph_m = statistics.fmean(phs)
        d_m, lo, hi = paired_bootstrap_ci(deltas, n_boot=n_boot)
        print(f"    n={len(deltas):>3d}  5uwbvhqf={prior_m:.4f}  Phase 2={ph_m:.4f}  delta={d_m:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=list(JUDGE_FILES.keys()) + ["all"], default="all")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    judges = list(JUDGE_FILES.keys()) if args.judge == "all" else [args.judge]
    for j in judges:
        prior_path, matched_path = JUDGE_FILES[j]
        if not Path(prior_path).exists() or not Path(matched_path).exists():
            print(f"\nJudge {j}: SKIP — files not yet present")
            continue
        report_one_judge(j, prior_path, matched_path, args.n_boot)


if __name__ == "__main__":
    main()
