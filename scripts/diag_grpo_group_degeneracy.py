"""Phase 2 — GRPO group-advantage degeneracy from train-time rollout dump (H2).

Reads the verl rollout dump (one JSONL per train step, each line one rollout
with `input` and `score` and `chosen_score`/`rejected_score`). Groups rollouts
by prompt and reports:

  - % of groups where all n rollouts received the same reward (advantage=0)
  - % of groups with binary {0,1} variation (informative)
  - distribution of within-group reward std
  - distribution of chosen-rejected score gaps (continuous signal)

If degenerate-group fraction > 40%, GRPO is starved of gradient on most groups
and that's a primary cause of the observed train-eval mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def _pstdev(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if len(xs) < 2: return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _quantiles(xs, n=10):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if len(xs) < n: return None
    out = []
    for i in range(1, n):
        idx = int(round(i / n * (len(xs) - 1)))
        out.append(xs[idx])
    return out


def load_step(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def analyze_step(rows: list[dict], expected_n: int) -> dict:
    by_input = defaultdict(list)
    for r in rows:
        by_input[r.get("input", "")].append(r)

    n_groups = len(by_input)
    n_groups_full = sum(1 for g in by_input.values() if len(g) == expected_n)
    n_degenerate = 0
    n_all_one = 0
    n_all_zero = 0
    n_mixed = 0
    stds = []
    margins = []
    rewards_means = []
    rubric_parse_rates = []

    for inp, group in by_input.items():
        scores = [float(r.get("score", 0.0)) for r in group]
        rewards_means.append(_mean(scores))
        if len(set(scores)) == 1:
            n_degenerate += 1
            if scores[0] == 1.0:
                n_all_one += 1
            elif scores[0] == 0.0:
                n_all_zero += 1
        else:
            n_mixed += 1
        if len(scores) > 1:
            stds.append(_pstdev(scores))
        # chosen-rejected margin per rollout
        for r in group:
            cs, rs = r.get("chosen_score"), r.get("rejected_score")
            if cs is not None and rs is not None:
                margins.append(float(cs) - float(rs))
        rubric_parse_rates.append(
            _mean(1.0 if r.get("rubric_parse_success", False) else 0.0
                             for r in group))

    out = {
        "n_groups": n_groups,
        "n_groups_full": n_groups_full,
        "expected_n": expected_n,
        "n_degenerate": n_degenerate,
        "n_all_one": n_all_one,
        "n_all_zero": n_all_zero,
        "n_mixed": n_mixed,
        "frac_degenerate": n_degenerate / n_groups if n_groups else None,
        "frac_all_one": n_all_one / n_groups if n_groups else None,
        "frac_all_zero": n_all_zero / n_groups if n_groups else None,
        "within_group_std_mean": _mean(stds) if stds else None,
        "within_group_std_pctiles": (
            [_quantiles(stds, n=10)[i] for i in (0, 4, 8)]
            if _quantiles(stds, n=10) else None),
        "margin_mean": _mean(margins) if margins else None,
        "margin_stdev": _pstdev(margins) if len(margins) > 1 else None,
        "margin_pctiles": (
            [_quantiles(margins, n=10)[i] for i in (0, 4, 8)]
            if _quantiles(margins, n=10) else None),
        "rubric_parse_rate_mean": _mean(rubric_parse_rates) if rubric_parse_rates else None,
        "step_mean_reward": _mean(rewards_means) if rewards_means else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--rollouts-per-prompt", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rdir = Path(args.rollout_dir)
    if not rdir.exists():
        raise SystemExit(f"rollout dir not found: {rdir}")

    step_files = sorted(rdir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    print(f"found {len(step_files)} step files in {rdir}")
    per_step = {}
    for f in step_files:
        try:
            step = int(f.stem)
        except ValueError:
            continue
        rows = load_step(f)
        if not rows:
            continue
        per_step[step] = analyze_step(rows, args.rollouts_per_prompt)
        per_step[step]["n_rollouts"] = len(rows)
        print(f"  step {step}: groups={per_step[step]['n_groups']:3d}  "
              f"frac_degen={per_step[step]['frac_degenerate'] or 0:.2f}  "
              f"frac_all_1={per_step[step]['frac_all_one'] or 0:.2f}  "
              f"reward={per_step[step]['step_mean_reward'] or 0:.3f}  "
              f"margin={per_step[step]['margin_mean'] or 0:.2f}")

    # Aggregate across steps
    fracs_degen = [v["frac_degenerate"] for v in per_step.values() if v["frac_degenerate"] is not None]
    fracs_all_one = [v["frac_all_one"] for v in per_step.values() if v["frac_all_one"] is not None]
    margins_mean = [v["margin_mean"] for v in per_step.values() if v["margin_mean"] is not None]

    headline = {
        "n_steps_analyzed": len(per_step),
        "frac_degenerate_mean": _mean(fracs_degen) if fracs_degen else None,
        "frac_degenerate_max": max(fracs_degen) if fracs_degen else None,
        "frac_all_one_mean": _mean(fracs_all_one) if fracs_all_one else None,
        "frac_all_one_max": max(fracs_all_one) if fracs_all_one else None,
        "margin_mean_across_steps": _mean(margins_mean) if margins_mean else None,
        "verdict": None,
    }
    fd = headline["frac_degenerate_mean"] or 0
    if fd > 0.5:
        headline["verdict"] = f"H2 STRONGLY CONFIRMED — {fd*100:.0f}% of groups have zero advantage. Sparse reward × n=4 is starving the gradient."
    elif fd > 0.3:
        headline["verdict"] = f"H2 PARTIALLY CONFIRMED — {fd*100:.0f}% degenerate groups. Significant signal loss but not the only cause."
    else:
        headline["verdict"] = f"H2 RULED OUT — only {fd*100:.0f}% degenerate groups. Sparse reward isn't the dominant pathology."

    out = {"headline": headline, "per_step": per_step}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nHEADLINE:")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
