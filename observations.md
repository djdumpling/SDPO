# SDPO RuPO LitBench — Observations

State as of 2026-05-04, ~17:00 UTC. Phase 5 cross-judges still in flight; this doc will be appended to once they land.

## TL;DR

1. **Reward-shape ablation (May 3, n=1 per shape, cosine LR, 200 steps):** `criteria_total_absolute` peaks at val/reward 0.700 vs 0.60-0.68 for the other four shapes. **Confirmed independently by Phase 5** (see point 5).
2. **Constant LR fixes the cosine late-training decay** (5uwbvhqf at constant LR finished at 0.6917 vs cosine baseline 0.5958 with the same shape).
3. **Across-seed variance is large** (final val/reward seed=1: 0.6917, seed=2: 0.6258, seed=3: 0.3779 with structural val pathology). The strongest run is the first one we trained.
4. **Cross-judge analysis is mostly inconclusive.** The earlier "GPT-5 cross-judge prefers Phase 2" finding was a sampling/staging artifact. On matched prompts and training steps, GPT-5 says 5uwbvhqf and Phase 2 are tied; Sonnet 4.6 says 5uwbvhqf wins by 12 pp (borderline-significant). Phase 2 shows no reward-hacking signature, but also no held-out advantage.
5. **Phase 5 (criteria_margin + constant LR + seed=1) finished at 0.5950**, well below `total_absolute` at the same schedule (5uwbvhqf 0.6917, Phase 2 0.6258). **The reward shape, not the LR schedule, is the real driver of `total_absolute`'s advantage.**

## 1. Reward-shape ablation (May 3 — 200-step cosine, seed=42)

All identical hyperparameters except the reward shape. Self-judge (Qwen3-8B) val/reward, n=819 per evaluation.

| wandb id | Reward shape | Peak val/reward | Peak step | Final val/reward | Final step |
|---|---|---:|---:|---:|---:|
| 4qn6qeob | `margin` (legacy non-criteria) | 0.6807 | 50 | 0.6789 | 200 |
| eyzlvijs | `criteria_margin` | 0.6011 | 100 | 0.5932 | 200 |
| qrychp86 | `criteria_margin_bounded` (dz unknown) | 0.6179 | 50 | 0.5656 | 200 |
| tj0d7580 | `criteria_margin_bounded` (dz unknown) | 0.6204 | 50 | 0.5914 | 200 |
| nybpt1hp | `criteria_total_absolute` | **0.6996** | 50 | 0.5958 | 200 |

**Observations:**
- `criteria_total_absolute` is the clear peak winner (~7-9 pp above the other criteria-based shapes).
- All five runs peak around step 50 and decay through step 200 under cosine LR. This is what motivated the constant-LR follow-up.
- `criteria_margin_bounded` runs (qrychp86 and tj0d7580) cannot be distinguished into dz=2 vs dz=5 from wandb config alone — the run names just say "manual."

## 2. Constant-LR follow-up — single seed (May 4, 100-step constant)

Same `criteria_total_absolute` shape with constant LR (no decay).

| wandb id | Config | Peak val/reward | Peak step | Final val/reward | Final step |
|---|---|---:|---:|---:|---:|
| 5uwbvhqf | constant LR, seed=1, 100 steps | **0.7027** | 75 | **0.6917** | 100 |

**Observations:**
- Constant LR holds the peak past step 50; final val/reward 0.6917 is +10 pp above the cosine baseline at the same shape.
- This is the headline single-run result — the strongest val/reward we have.

## 3. Overnight multi-seed + disconfound (May 4)

Five-phase orchestration. Phases 1, 3, 6 are evaluation phases; phases 2, 4, 5 are training runs.

| Phase | wandb id | Config | Peak | Peak step | Final | Final step | Notes |
|---|---|---|---:|---:|---:|---:|---|
| 2 | 2a9wdgln | constant LR, seed=2, total_absolute | 0.6819 | 50 | 0.6258 | 100 | clean |
| 4 | 4nk2l4fm | constant LR, seed=3, total_absolute | 0.6917 | 25 | **0.3779** | 100 | val pathology (see below) |
| 5 (orig.) | rv9mxl8u | constant LR, seed=1, **`criteria_margin`** | 0.5866 | 25 | 0.5786 | 50 | crashed (vLLM OOM @ step 50 wake-up) |
| 5 (rerun) | mx2iljoe | same, with reduced rollout-mem | 0.5950 | 100 | **0.5950** | 100 | clean |

### Multi-seed (constant LR + total_absolute)

| Seed | wandb id | Final val/reward |
|---|---|---:|
| 1 | 5uwbvhqf | 0.6917 |
| 2 | 2a9wdgln | 0.6258 |
| 3 | 4nk2l4fm | 0.3779 (degenerate) |
| **mean (n=2, excl. seed=3)** | | **0.6588** |
| **mean (n=3)** | | 0.5651 |

### Seed=3 caveat

At step 100 val: `judge_called/mean@1 = 0.564` vs training-time `judge_called = 0.917`. The 35 pp drop is structural rubric-validity failure on val outputs (early-exit in `reward_fn.py` lines 493-504 / 386). Phase 4 was launched with `VALIDATION_DATA_DIR=null` so the malformed val outputs are not on disk for inspection. Treating seed=3 as an outlier; n=2 reported pending a follow-up val-output capture run.

### Phase 5 — disconfound conclusion

Phase 5 isolates the question "is `total_absolute` actually a better shape, or did it just happen to peak before the cosine decay started?" by training `criteria_margin` at constant LR (the same schedule as 5uwbvhqf).

- Phase 5 final: 0.5950 at step 100
- Compare to `criteria_margin` under cosine LR (eyzlvijs): 0.5932 final at step 200 — basically equal
- Compare to `total_absolute` under constant LR (5uwbvhqf): 0.6917 — **+10 pp**

**Conclusion:** `criteria_margin` does not benefit from constant LR the way `total_absolute` does. The reward shape is the real driver of `total_absolute`'s advantage, not the schedule. The 5-shape ablation conclusion survives the disconfound.

## 4. Cross-judge analysis (the most nuanced piece)

The training-time judge is the same Qwen3-8B family as the policy, so self-judge val/reward can be inflated by judge-specific gaming (Goodhart). Cross-judging with held-out models is the de-confound.

### How the interpretation evolved

| Stage | What we thought | What corrected it |
|---|---|---|
| Initial (n=80 5uwbvhqf, n=192 Phase 2 GPT-5) | "GPT-5 prefers Phase 2 (0.732) over 5uwbvhqf (0.638) — ranking flips under a held-out judge, suggestive of robustness" | n=192 was the orchestrator's default cap on the first 200 val-parquet rows — a non-random, upward-biased subset (self-judge on those 192 was 0.685 vs full-set 0.626) |
| After GPT-5 n=819 | "Phase 2 cross-judge 0.664 (down from 0.732) — gap with self-judge effectively zero, no robustness gain" | Apples-to-oranges: Phase 2 was being judged at step 100 (its final), 5uwbvhqf only at steps 25/50 (the only ones in its wandb table) |
| After matched 20-prompt × 2-step comparison | "5uwbvhqf and Phase 2 are tied under GPT-5; Sonnet favors 5uwbvhqf by 12 pp" | This is the cleanest read — same prompts, same training stages |

### Headline numbers (with bootstrap 95% CIs, 5000 resamples, paired)

**Within-run self − cross-judge gap:**

| Run | n | self − GPT-5 gap | self − Sonnet gap |
|---|---:|---|---|
| 5uwbvhqf (wandb subset) | 80 | **+0.131 [+0.025, +0.238]** ✓ | 0.000 [-0.100, +0.100] |
| Phase 2 (full set) | ~760 | +0.005 [-0.026, +0.036] | -0.019 [-0.050, +0.012] |
| Phase 2 (matched 20 × 2 steps) | 40 | +0.025 [-0.113, +0.163] | +0.038 [-0.113, +0.200] |

**Only one statistically significant gap: 5uwbvhqf under GPT-5.** Sonnet doesn't corroborate. Phase 2 has no significant gap under either external judge.

**Head-to-head 5uwbvhqf vs Phase 2 on matched 20 prompts × 2 steps (clean comparison):**

| Judge | 5uwbvhqf | Phase 2 | Δ (5u − Ph2) | 95% CI |
|---|---:|---:|---:|---|
| GPT-5 (pooled) | 0.6583 | 0.6625 | -0.004 | [-0.171, +0.163] (tied) |
| Sonnet (pooled) | 0.7708 | 0.6500 | **+0.121** | **[+0.000, +0.267]** (5uwbvhqf wins, borderline) |

### Defensible claims

- **5uwbvhqf has at least some self-judge-specific lift, detected by GPT-5 but not Sonnet.** One-of-two held-out judges supports the inflation hypothesis. Suggestive, not conclusive.
- **Phase 2 shows no reward-hacking signature.** Both held-out judges agree with self-judge to within ~2 pp on the full set.
- **Phase 2 is *not* clearly better than 5uwbvhqf under held-out evaluation.** GPT-5 ties them; Sonnet favors 5uwbvhqf. The original "Phase 2 wins" headline collapsed under proper sampling.

### Methodological lessons (worth preserving)

- **Never report cross-judge on a non-random subset** without verifying the subset's self-judge tracks the full set. The n=192 vs n=819 GPT-5 gap (0.732 → 0.664) was entirely due to the n=192 being the first 200 rows, which were upward-biased.
- **Match training stages when comparing across runs.** 5uwbvhqf-step-50 vs Phase-2-step-100 is two factors mixed into one number.
- **The 80-row wandb subset for 5uwbvhqf was upward-biased** by ~+8 pp (self-judge 0.7688 vs full-set 0.6917). When the full-set was inaccessible (no checkpoints, no JSONL dumps for 5uwbvhqf), apples-to-apples required filtering Phase 2 to those same 20 unique prompts at the same steps.

## 5. Phase 5 cross-judges (in flight)

`criteria_margin + constant LR + seed=1` cross-judged with both GPT-5 and Sonnet 4.6 at n=819. ETA ~30-60 minutes from launch. Will be appended to this file once they land.

The interesting question: does Phase 5 also show the consistent zero-gap pattern that Phase 2 has? If so, the "rubrics are judge-portable across reward shapes" story is broader than total_absolute. If Phase 5 shows a positive gap, then the zero-gap is specific to total_absolute (more interesting).

## 6. Open issues for the writeup

- **n=2 multi-seed** is the soft spot. Across-seed variance is large (0.626 to 0.692 across the two clean seeds; 0.378 if we include the degenerate seed=3). One more seed would tighten the variance estimate.
- **Seed=3's val pathology is uncharacterized.** No JSONL dumps; the failure mode (rubrics structurally invalid only at val time) is hypothesized but not visually verified.
- **Self-judge inflation evidence is single-judge.** GPT-5 detects it; Sonnet doesn't. A third judge family (Gemini Pro 2.5 has been spot-checked as available on pinference) would break the tie. Optional, ~$30-50.
- **The original 5-shape ablation conclusion was made on self-judge only.** Cross-judging those May-3 runs would require retraining each shape with `VALIDATION_DATA_DIR` enabled (the original runs predate that orchestration). High effort, low marginal value given Phase 5 already confirms the shape ranking.

## 7. Cost summary

- Phase 1 + Phase 3 cross-judges (overnight): GPT-5 only, ~$30-40
- Sonnet 4.6 second cross-judge (n=80 + n=819): ~$25
- GPT-5 n=819 rerun: ~$45
- Matched 20-prompt cross-judges (both judges): ~$15
- Phase 5 cross-judges (in flight, both judges, n=819): ~$50-70 estimated

**Total: ~$165-195** at projected completion. Single human evaluation pass on n=200 stories at typical research rates would have been $2-9k for inferior coverage, so the spend is reasonable.

## File pointers

- All cross-judge JSONs: `logs/seed2_constantlr_total_absolute_20260504-071351/crossjudge_*.json`
- Bootstrap CI script: `scripts/bootstrap_crossjudge_cis.py`
- Matched-subset comparison: `scripts/compare_matched_crossjudge.py`
- Matched-subset filter: `scripts/build_matched_subset.py`
- Second-judge driver: `scripts/run_second_crossjudge.sh`
- Original overnight orchestrator (with results in SUMMARY.md): `run_overnight_seed2_and_crossjudge.sh`
- Phase 5 rerun training log: `logs/phase5_rerun_criteriamargin_20260504-150016/training.log`
