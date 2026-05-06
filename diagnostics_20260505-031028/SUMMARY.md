# Overnight diagnostics — diagnostics_20260505-031028

_Generated 2026-05-05T09:08:29_

## Phase status (rc per phase)

```
p1_prompt_mode_audit 0
p2_grpo_group_degeneracy 0
p3_judge_test_retest 0
p4_litbench_trained_k1 0
p4_litbench_trained_k2 0
p4_litbench_trained_k3 0
p4_litbench_trained_k4 0
p4_litbench_trained_aggregate 1
p4_litbench_baseline_k1 0
p4_litbench_baseline_k2 0
p4_litbench_baseline_k3 0
p4_litbench_baseline_k4 0
p4_litbench_baseline_aggregate 0
p5_gpt5_litbench_baseline 0
p5_gpt5_hs3_baseline 143
p5_gpt5_arena_baseline 143
p6_litbench_trained_flipped 0
p7_minimal_prompt_litbench 0
```

## Phase 1 — Prompt-mode audit (H8)

- POSSIBLE — preprocess routes by is_eval/split; manual inspection required
- Open preprocess_dataset.py around the `policy_prompt_mode` line and confirm both train and val rows go through the same code path. The untrained baseline uses `policy_prompt_mode=prompt_only`; the trained policy uses `pair`. If val for the trained policy uses `pair` mode (sees both responses), and the untrained baseline uses `prompt_only`, the comparison is asymmetric.

## Phase 2 — GRPO group-advantage degeneracy (H2)

- Steps analyzed: 100
- Mean fraction of degenerate groups (all-same reward): **65.6%**
- Max fraction degenerate across steps: 91.7%
- Fraction all-1.0 (Goodhart success): 42.8%
- Mean chosen-rejected margin across steps: 7.062721592382112
- **Verdict:** H2 STRONGLY CONFIRMED — 66% of groups have zero advantage. Sparse reward × n=4 is starving the gradient.

## Phase 3 — Judge test-retest (H1)

- N triples: 100, K resamples: 5
- Judge per-call σ (test-retest, on margin): **9.53pt**
- Between-triple margin σ (between-rubric signal): 29.48pt
- Between-triple mean margin: 4.93pt
- SNR (mean margin / per-call σ): 0.5174633721853368
- **Verdict:** H1 CONFIRMED — judge per-call σ on margin = 9.5pt, between-triple mean margin = 4.9pt → SNR = 0.52. Judge noise dominates the rubric signal. k=4 ensembling should reduce σ to ~4.8pt and restore signal.

## Phase 4 — Judge ensembling re-eval (H1 fix)

### litbench_baseline_ensembled
- K=4  n_rows=298
- Individual-run accuracies: [0.7013422818791947, 0.6952861952861953, 0.702020202020202, 0.7315436241610739]
- Avg individual: **0.7075480758366665**
- Ensemble accuracy: **0.7164429530201343**
- Ensembling lift: **0.889487718346782 pp**
- Verdict: H1 marginal — k=4 lift = 0.9pp. Ensembling helps a little; not the dominant fix.

### litbench_trained_ensembled
- K=4  n_rows=286
- Individual-run accuracies: [0.708041958041958, 0.6835664335664335, 0.7062937062937062, 0.6748251748251748]
- Avg individual: **0.6931818181818181**
- Ensemble accuracy: **0.6958041958041958**
- Ensembling lift: **0.2622377622377714 pp**
- Verdict: H1 marginal — k=4 lift = 0.3pp. Ensembling helps a little; not the dominant fix.


## Phase 5 — GPT-5 cross-judge on UNTRAINED baselines (H3)

- **litbench_baseline_gpt5**: GPT-5 accuracy = 0.7613 on n=398

Compare to existing trained-model GPT-5 results in `logs/.../crossjudge_*_gpt5*.json` to see whether GPT-5 reveals a trained-vs-baseline gap (H3 falsifier).

## Phase 6 — Position-shuffle eval (H7)

- Flipped accuracy: **0.30666666666666664** on n=300
- Verdict: H7 RULED OUT — flipped accuracy = 0.307 ≈ 1 - original_accuracy. Judge correctly identifies quality regardless of position; no position bias.

## Phase 7 — Minimal-prompt baseline (H11)

- Minimal-prompt accuracy: **None** (n_parsed=0/400)
- Verdict: Could not parse responses; check model/server.

## Cross-cutting interpretation

Pull the verdicts above and decide which scenario applies:

- **Scenario A (judge-noise + low rollouts)**: H1 confirmed (Phase 3 SNR<1) + H2 confirmed (Phase 2 frac_degen>40%) + Phase 4 ensembling lift ≥1pp. → Run conditional retrain #12 (n=8 rollouts, judge ensembled k=2-4, std-norm GRPO, LR=1e-5, rubric max=1024).
- **Scenario B (judge ceiling + label noise)**: Phase 5 GPT-5 baselines do NOT exceed ~0.72 → judge ceiling is binding. Don't retrain; reframe paper around cross-judge shift + interpretability.
- **Scenario C (eval bug)**: Phase 1 reveals train/val mode mismatch, OR Phase 6 reveals position bias. → Bug-fix and re-run.

Update `05_04_26_research_plan.md` and `observations_v2.md` with the verdict.
