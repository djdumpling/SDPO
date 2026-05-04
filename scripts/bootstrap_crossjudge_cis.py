"""Bootstrap 95% CIs for every cross-judge result in a log dir.

Reads each crossjudge_*.json in the dir, recomputes mean_crossjudge_reward and
mean_original_reward with bootstrap CIs (default 5000 resamples) at the row
level. Also reports the within-run gap (self − cross) with its CI, since that
is the cleanest over-optimization signal.

Usage:
  python scripts/bootstrap_crossjudge_cis.py [--log-dir <dir>] [--n-boot 5000]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

DEFAULT_LOG_DIR = "/root/rubric-policy-optimization/SDPO/logs/seed2_constantlr_total_absolute_20260504-071351"


def bootstrap_ci(values: list[float], n_boot: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lo, hi) where (lo, hi) is the 1-alpha bootstrap CI."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    n = len(values)
    rng = random.Random(0)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return statistics.fmean(values), lo, hi


def paired_gap_ci(self_vals: list[float], cross_vals: list[float], n_boot: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap the *paired* gap (self - cross) using the same row indices."""
    assert len(self_vals) == len(cross_vals)
    n = len(self_vals)
    rng = random.Random(1)
    diffs = [s - c for s, c in zip(self_vals, cross_vals)]
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return statistics.fmean(diffs), means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    files = sorted(log_dir.glob("crossjudge_*.json"))
    if not files:
        raise SystemExit(f"No crossjudge_*.json files in {log_dir}")

    print(f"{'file':75s} {'n':>5s} {'cross-judge':>20s} {'self-judge':>20s} {'self-cross gap':>20s}")
    print("-" * 145)
    for f in files:
        d = json.load(open(f))
        rows = [r for r in d["rows"] if r.get("judge_called") and not r.get("judge_error") and "reward" in r and r.get("original_score") is not None]
        cross_vals = [float(r["reward"]) for r in rows]
        self_vals = [float(r["original_score"]) for r in rows if r.get("original_score") is not None]
        # Re-pair so they line up by row order.
        paired = [(float(r["reward"]), float(r["original_score"])) for r in rows if r.get("original_score") is not None]
        cross_paired = [c for c, _ in paired]
        self_paired = [s for _, s in paired]
        cross_m, cross_lo, cross_hi = bootstrap_ci(cross_paired, args.n_boot)
        self_m, self_lo, self_hi = bootstrap_ci(self_paired, args.n_boot)
        gap_m, gap_lo, gap_hi = paired_gap_ci(self_paired, cross_paired, args.n_boot)
        print(f"{f.name:75s} {len(paired):>5d} "
              f"{cross_m:.4f} [{cross_lo:.4f},{cross_hi:.4f}] "
              f"{self_m:.4f} [{self_lo:.4f},{self_hi:.4f}] "
              f"{gap_m:+.4f} [{gap_lo:+.4f},{gap_hi:+.4f}]")


if __name__ == "__main__":
    main()
