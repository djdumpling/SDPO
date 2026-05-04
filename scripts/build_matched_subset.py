"""Build a matched-prompts cross-judge subset for apples-to-apples comparison.

5uwbvhqf's wandb table only captured 80 rollouts (20 unique prompts × 3 rollouts at
step 25 + 1 rollout at step 50). To compare directly against Phase 2 on the same
prompts at the same training stages, this script:

  1. Reads the prior cross-judge JSON to get the 20 unique prompts.
  2. Filters Phase 2's val_generations/{25,50}.jsonl to only those 20 prompts
     (1 row per prompt per step).
  3. Writes a new directory of filtered JSONLs that cross_judge_rubric.py
     can consume with --source jsonl --jsonl-step all.

Output structure (filterable by step):
  <out-dir>/25.jsonl  — 20 Phase 2 rows at step 25 matching prior prompts
  <out-dir>/50.jsonl  — 20 Phase 2 rows at step 50 matching prior prompts

Usage:
  python scripts/build_matched_subset.py
      --prior-crossjudge logs/.../crossjudge_prior_5uwbvhqf_gpt5.json
      --phase2-val-dir logs/.../val_generations
      --out-dir logs/.../matched20_val_generations
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-crossjudge", required=True,
                    help="Any one of the prior cross-judge JSONs (gpt5 or sonnet — they share the 20 prompts).")
    ap.add_argument("--phase2-val-dir", required=True,
                    help="Phase 2 val_generations/ directory containing {25,50,75,100}.jsonl.")
    ap.add_argument("--out-dir", required=True, help="Where to write the filtered JSONLs.")
    ap.add_argument("--steps", nargs="+", default=["25", "50"], help="Step files to filter (must exist in --phase2-val-dir).")
    args = ap.parse_args()

    prior = json.load(open(args.prior_crossjudge))
    prior_rows = prior["rows"]
    # The wandb loader's prompt extraction kept some trailing chat-template
    # scaffold (e.g. "\n\nHere is one candidate response:") that the verl JSONL
    # loader strips. Truncate at the first occurrence of any known scaffold
    # marker so matches succeed.
    SCAFFOLD_MARKERS = ["\n\nHere is one candidate response", "\n\nHere is the prompt", "\n<candidate_a>"]
    def normalize(p: str) -> str:
        s = p.strip()
        for m in SCAFFOLD_MARKERS:
            i = s.find(m)
            if i != -1:
                s = s[:i]
        return s.strip()
    prior_prompts = sorted(set(normalize(r["prompt"]) for r in prior_rows if r.get("prompt")))
    print(f"prior cross-judge: {len(prior_rows)} rows, {len(prior_prompts)} unique prompts (after scaffold-strip)")

    # Per-prompt step distribution in prior, for sanity printing.
    by_prompt_step = defaultdict(lambda: defaultdict(int))
    for r in prior_rows:
        by_prompt_step[normalize(r["prompt"])][r.get("step")] += 1
    step_dist = defaultdict(int)
    for p in prior_prompts:
        for s, c in by_prompt_step[p].items():
            step_dist[s] += c
    print(f"prior step distribution (rollouts × prompts): {dict(step_dist)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dir = Path(args.phase2_val_dir)
    for step in args.steps:
        src = val_dir / f"{step}.jsonl"
        if not src.exists():
            print(f"  step {step}: SKIP — {src} missing")
            continue
        kept = []
        seen = set()
        with src.open() as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gts_raw = d.get("gts")
                if not isinstance(gts_raw, str): continue
                try:
                    gt = json.loads(gts_raw)
                except json.JSONDecodeError:
                    continue
                p = str(gt.get("prompt", "")).strip()
                if p in prior_prompts and p not in seen:
                    seen.add(p)
                    kept.append(line.rstrip("\n"))
        out = out_dir / f"{step}.jsonl"
        out.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"  step {step}: matched {len(kept)}/{len(prior_prompts)} prior prompts → {out}")
        missing = [p for p in prior_prompts if p not in seen]
        if missing:
            print(f"    {len(missing)} prior prompts not found in Phase 2 step {step} (substring snippets):")
            for p in missing[:3]:
                print(f"      '{p[:80]}...'")


if __name__ == "__main__":
    main()
