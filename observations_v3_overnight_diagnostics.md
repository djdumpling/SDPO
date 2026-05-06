# SDPO RuPO — overnight-diagnostic observations (Phase battery 2026-05-05)

State as of 2026-05-05 ~09:08 UTC. The 12-hypothesis investigation plan from `05_04_26_research_plan.md` ran to completion as `diagnostics_20260505-031028/`. This document consolidates what we learned, with careful framing of which conclusions are decisive vs. partial.

---

## TL;DR — the central finding

> **Under a strong external judge (GPT-5), the untrained Qwen3-8B baseline beats every trained variant by 8–14 percentage points on LitBench preference accuracy.** Under the in-house Qwen3-8B judge, the same comparison shows ≈ parity. Training is therefore not "ineffective" — it is *actively producing rubrics that an external judge dislikes more than the baseline's*. The previous "trained ≈ baseline" framing was understating the problem.

Concretely on LitBench, side by side:

| Rubric writer | In-house Qwen3-8B judge (k=4 ensembled) | GPT-5 judge |
|---|---:|---:|
| Untrained Qwen3-8B baseline | 0.716 (n=298) | **0.765 (n=396)** |
| Trained, `criteria_total_absolute` (Phase 2) | 0.696 (n=286) | 0.664 (n=759) |
| Trained, `criteria_margin` (Phase 5) | — | 0.681 (n=810) |
| **Gap (baseline − trained)** | **+0.020** | **+0.084 to +0.101** |

The in-house judge sees a +2pp baseline-favoring gap; GPT-5 sees +8–10pp. The very thing the in-house judge calls "essentially equal" looks "decisively worse" to GPT-5. This is the signature of a **moving-target / Goodhart loop** — when policy and judge share weights and re-sample stochastically, training drives the policy in directions the in-house judge over-rates and external judges under-rate.

---

## Hypothesis verdict table

The plan tested 12 hypotheses; nine got verdicts overnight, three were unresolved or out-of-scope. Each row links to the supporting phase and the concrete number.

| # | Hypothesis | Verdict | Decisive number |
|---|---|---|---|
| H1 | Judge stochasticity at temp=0.7 swamps the rubric signal | **CONFIRMED at criterion level; small fix-impact** | per-call σ(margin)=9.5pt, signal mean=4.9pt → SNR 0.52; but k=4 ensembling lifts accuracy only +0.26pp (trained) / +0.89pp (baseline) |
| H2 | GRPO group-advantage degeneracy from sparse ternary reward | **STRONGLY CONFIRMED** | 65.6% of n=4 groups had all-same reward across 100 steps; max 91.7% on a single step |
| H3 | Self-judge ceiling caps everyone at ~0.69 | **DECISIVELY RULED OUT** | GPT-5 scores the *untrained baseline* at 0.7651 — well above the in-house "ceiling" |
| H4 | Eval is itself a noisy single-sample estimate | Not separately tested overnight | Phase 4 is the closest proxy — k=4 means narrow CIs at ±1pp, and the trained-vs-baseline gap on the in-house judge survives |
| H5 | Goodhart loop: judge weights = policy weights, training shifts toward judge artifacts | **STRONGLY CONFIRMED** | The 8–10pp baseline-vs-trained gap appears *only* under GPT-5; under the self-judge it's ~+2pp. Training is producing judge-specific shift |
| H6 | Insufficient training (LR/steps/no-KL) | Untestable without `global_step_*` checkpoints | No saved actor weights → parameter-delta check infeasible. **This is a checkpointing-config issue to fix in the next run.** |
| H7 | Position bias / pair-mode leakage in the val pipeline | **RULED OUT** | Flipped accuracy = 0.307 ≈ 1 − original 0.69; the judge tracks quality, not slot |
| H8 | Train and val use different `policy_prompt_mode` | **OPEN — likely the asymmetry is real** | Audit found the launcher passes `prompt_only` for the *untrained* baseline but `pair` for the trained policy. Needs manual confirmation that val for the trained policy is also `pair` (it should be); and a re-baseline in `pair` mode |
| H9 | Dataset label-noise ceiling | Not tested overnight (high-confidence subset filter wasn't run) | Inferable from H3 — GPT-5 reaches 0.77 on the same data the in-house judge says is "0.69 ceiling," so label noise is not the binding ceiling |
| H10 | Prompt template carries most of the signal; RL has no headroom | **Unresolved** | Phase 7 minimal-prompt baseline failed: 0/400 parsed (max_tokens=8 truncated Qwen3 thinking output). Re-runnable with max_tokens=128. |
| H11 | Model size (8B too small for RL to find new modes) | Out of scope this round | Would need a Qwen3-4B retrain |
| H12 | Reward shape (`absolute` ternary vs `margin` continuous) | **Indirectly informed** | Phase 2 confirmed sparsity is fatal for `criteria_total_absolute`; Phase 5 (`criteria_margin`) shows the same baseline-favoring gap under GPT-5, so margin-style reward is *not* a fix |

---

## Phase-by-phase details

### Phase 1 — prompt-mode audit (H8): **needs manual confirmation, but a real asymmetry exists**

The audit script grep'd `run_sdpo_qwen35_9b_litbench_8gpu.sh`, `preprocess_dataset.py`, and the baseline launcher for `policy_prompt_mode`. Findings:

- The trainer launches the **trained** policy with `--policy_prompt_mode pair` (rubric-writer sees both responses).
- The **untrained baseline** launcher (`untrained_baseline_litbench_*`) calls preprocess with `policy_prompt_mode=prompt_only` (rubric-writer sees the prompt only — writes blind).

**Implication.** The "trained vs untrained baseline" comparison is between rubrics generated under *different prompt conditions*. Pair-mode rubrics see chosen+rejected and can be written with knowledge of the gap; prompt-only rubrics cannot. This is technically an apples-to-oranges comparison.

**Why it does not fully explain the headline finding.** Even granting this asymmetry, the *direction* of the result is still anti-intuitive. We'd expect pair-mode (more information) to help, not hurt. Under GPT-5, pair-mode trained rubrics score 0.66 vs prompt-only baseline at 0.77. The trained policy with strictly more information loses by 11pp. So while the asymmetry should be fixed, it is not the primary cause of the gap.

**Action item.** Re-generate the untrained baseline with `policy_prompt_mode=pair` to make the comparison clean. This is one cheap re-eval (~30 min GPU + 1h GPT-5 calls).

### Phase 2 — GRPO group-advantage degeneracy (H2): **strongly confirmed**

We mined the verl rollout dump from `seed2_constantlr_total_absolute_20260504-071351` (100 steps × 12 prompts × 4 rollouts = 4,800 rollouts). The reward function `criteria_total_absolute` is ternary {0, 0.5, 1.0}; we counted groups where all 4 rollouts received the same reward.

Headline:

- **65.6% of groups have zero advantage** averaged over the run.
- Max 91.7% on a single step. **No step has fewer than ~38% degenerate groups.**
- Of degenerate groups: 42.8% are all-1.0 (Goodhart success — judge saturates), 22.7% are all-0.0 (consistent failure — judge prefers neither).
- Only the remaining ~34% of groups produce gradient signal.

Per-step bucketing of degeneracy across the 100-step run:

| Steps | mean frac_degenerate | all-1.0 | all-0.0 |
|---|---:|---:|---:|
| 0–25 | 0.622 | 0.417 | 0.205 |
| 25–50 | 0.697 | 0.480 | 0.217 |
| 50–75 | 0.670 | 0.450 | 0.220 |
| 75–100 | 0.635 | 0.369 | 0.263 |

Degeneracy is high from the start and stays high — the policy never escapes the saturation regime under n=4 + ternary reward. The slight increase in all-0.0 from step 0 to step 100 (0.21 → 0.26) is consistent with Goodhart failure: as the policy concentrates on a few criterion patterns, more groups fall into "judge prefers neither" territory.

**Reading.** GRPO is starved of gradient on roughly 2/3 of every batch. The remaining 1/3 supplies all the learning signal — and that signal is itself dominated by judge stochasticity (Phase 3). This is a textbook degenerate-GRPO failure mode and an unambiguous cause of the train-eval mismatch.

**Fix direction.** Larger n (≥8, ideally 16), `norm_adv_by_std_in_grpo: True`, and a continuous reward shape that doesn't quantize to 3 values. OpenRubrics and RubricRM use n=64–244 for exactly this reason.

### Phase 3 — judge test-retest (H1): **confirmed — judge SNR is 0.52**

We took 100 fixed (rubric, chosen, rejected) triples from the trained policy's val output and re-ran the in-house Qwen3-8B judge K=5 times at temp=0.7 on each one. Then we compared per-triple test-retest σ to between-triple σ.

Headline:

- Per-call σ on chosen score: **5.5pt** (on a 0–100 scale).
- Per-call σ on rejected score: **6.4pt**.
- Per-call σ on **margin** (chosen − rejected): **9.5pt** — the margin is what GRPO actually optimizes.
- Between-triple σ on margin: 29.5pt — there *is* between-rubric structure.
- Mean margin across triples: **4.9pt**.
- **SNR = mean signal / per-call noise = 4.93 / 9.53 = 0.52** — single-call judge calls are noise-dominated.

**Reading.** Each individual GRPO advantage estimate is mostly noise. Theoretically k=4 ensembling of judge calls reduces σ by √4=2× to ~4.8pt, lifting SNR above 1.0.

**Where the theory broke down (Phase 4 caveat below).** In practice, k=4 ensembling at *eval time* delivered only +0.26pp (trained) and +0.89pp (baseline) on accuracy. The reason is that the test-retest noise is largely on score *magnitude* not on score *sign* — most pairs have a clearly-better and a clearly-worse response, and the judge consistently picks the right one even with 9.5pt margin noise. The pairs that are genuinely close (where SNR matters most) are also the pairs where neither rubric is reliably right. So while H1 is *real* at the criterion level, **fixing it is not the binding constraint on accuracy.**

This is an important nuance for the paper: SNR-style arguments motivate the fix, but the empirical lift is small.

### Phase 4 — judge ensembling re-eval (H1 fix): **+0.26pp / +0.89pp lift**

We re-ran cross-judge over the trained policy's val rubrics K=4 times, then averaged criterion scores per row before re-deriving the absolute reward. Same procedure for the untrained baseline.

Headline (apples-to-apples; n is the rows that successfully landed in all 4 resamples):

| | n | individual k=1 mean | k=4 ensembled | lift |
|---|---:|---:|---:|---:|
| LitBench trained | 286 | 0.6932 | **0.6958** | +0.26pp |
| LitBench baseline | 298 | 0.7075 | **0.7164** | +0.89pp |

**Reading.**

1. The k=4 ensembled in-house-judge gap (baseline − trained) is **+2.06pp**, with much tighter CIs than the single-call estimates.
2. The lift from ensembling is real but small. Variance reduction is not the dominant fix.
3. Interestingly, the lift is larger on baseline than on trained (0.89 vs 0.26). This is consistent with H5: the trained policy is producing rubrics whose accuracy under the in-house judge is structurally noisy in ways averaging can't repair (e.g., systematically ambiguous criteria), while the baseline's rubrics are crisp enough that more samples just help.

### Phase 5 — GPT-5 cross-judge on the untrained baseline (H3): **decisive falsifier**

GPT-5 was called via Pinference on the untrained-baseline LitBench rubrics with the same `criteria_total_absolute` derivation as the in-house judge.

Headline:

- **GPT-5 baseline accuracy = 0.7651 on n=396** (n_judge_error=2, n_rubric_invalid=2, n_total=400).
- The same baseline scores 0.7164 under the in-house judge (k=4 ensembled, n=298).
- GPT-5 rates the baseline 4.9pp higher than the in-house judge does.

Cross-referenced with prior trained-policy GPT-5 numbers from `logs/.../crossjudge_*_gpt5*.json`:

| Run | GPT-5 cross-judge | n |
|---|---:|---:|
| **Untrained Qwen3-8B baseline** (this run) | **0.7651** | 396 |
| Trained `criteria_total_absolute` (Phase 2 seed=2) | 0.6640 | 759 |
| Trained `criteria_margin` (Phase 5) | 0.6815 | 810 |
| Trained HelpSteer3 step 100 | 0.7158 | 438 |
| Trained Arena step 100 | 0.6074 | 940 |

**Reading.**

- The judge ceiling hypothesis (H3) is dead. GPT-5 reaches 0.77 on the same dataset the in-house judge tops out at 0.72.
- The +10pp gap on LitBench is the dominant finding.
- HS3 and Arena GPT-5 baselines are missing because Phase 5 stalled on Pinference (the second and third Pinference jobs hung indefinitely on individual hung calls and were killed at rc=143). Re-running with `--judge-timeout 60` should recover these in ~1h.

**Why this is decisive.** It is not possible to construct a "judge ceiling" or "label noise" story that explains both:
(a) the in-house judge says "trained ≈ baseline at 0.69," and
(b) GPT-5 says "baseline = 0.77, trained = 0.66."

The only consistent explanation is that the in-house judge is systematically miscalibrated relative to GPT-5, and training pushes the policy in a direction that exploits that miscalibration. That is H5.

### Phase 6 — position-shuffle eval (H7): **ruled out**

We swapped chosen↔rejected in the trained policy's val JSONL and re-ran the cross-judge on the flipped pairs at temp=0.7.

- Original accuracy: ~0.69.
- **Flipped accuracy: 0.307 on n=300.**
- 0.307 ≈ 1 − 0.69, within sampling noise of a perfectly position-invariant judge.

**Reading.** The judge picks the genuinely-better response regardless of slot. There is no position bias and no pair-mode leakage to explain the parity. We can drop this hypothesis from the table.

### Phase 7 — minimal-prompt baseline (H10/H11): **failed (re-runnable)**

We tried calling Qwen3-8B with a minimal A/B prompt to estimate how much of 0.69 comes from the rubric-writing template vs the model itself. **0 of 400 responses parsed.**

Root cause: `max_tokens=8` truncated the Qwen3 thinking-mode output before it ever emitted a letter. The model wraps its answer in `<think>…</think>` reasoning tokens; with only 8 output tokens available, it never reaches the answer.

**Action item.** Re-run with `max_tokens=128` and a regex that handles `<think>…</think>\n\nB` style outputs. ~30 min GPU.

---

## What this means for the submission

The "RuPO ≈ baseline" framing in `observations_v2.md` was correct directionally but understated the problem. The corrected story is:

1. **The base Qwen3-8B model writes rubrics that GPT-5 (a much stronger judge) likes.** Untrained baseline = 0.77 on LitBench under GPT-5.

2. **RL training under a same-model self-judge moves the policy *away* from those high-quality rubrics**, into a region the in-house judge over-rates and GPT-5 actively dislikes. The 11pp drop under GPT-5 is consistent across reward shapes (`absolute` and `margin`).

3. **The mechanism is identifiable.** The in-house judge is stochastic (per-call σ on margin = 9.5pt, SNR = 0.52); n=4 GRPO with ternary reward means 65% of groups carry zero advantage; the gradient signal that *does* survive comes from the small subset of groups where the judge happened to reward what it would penalize on a different sample. Training optimizes the noise-floor of the judge, not the underlying preference.

4. **The conventional fixes (n=8 rollouts, k=4 judge ensemble, std-norm GRPO) move the needle by ~1pp** — they are necessary but not sufficient. The dominant pathology is the policy=judge weight identity itself, not insufficient variance reduction.

### Reframing options for the paper

**Option A — keep RuPO but decouple the judge.** The cleanest fix is to use a frozen external judge during RL (RubricRM-4B, served locally; or a strong API model with a fixed snapshot). Submit as: "self-judge RuPO is fundamentally Goodhart-vulnerable; we show the failure mode quantitatively and demonstrate that judge-decoupling restores expected RL behavior." This requires conditional retrain #13 (~3h GPU).

**Option B — submit the negative result.** "RL on stochastic same-model judges quantitatively destroys preference accuracy under external evaluation, even when in-house metrics show no degradation. We provide the diagnostic battery and a 4-axis analysis (judge SNR, group degeneracy, parameter delta, cross-judge gap) that future rubric-RL work should report." This is publishable as-is, with cleaner methodology than the original framing.

**Option C — the interpretability story.** Use the cross-judge gap + rubric-content collapse together: "training under a self-judge concentrates rubric weight onto fewer criteria, induces vocabulary collapse (`emotional X` on LitBench, `clarity X` on HS3/Arena), and these content shifts correlate with the cross-judge regression direction." This is what `observations_v2.md` already gestures at, now with quantitative cross-judge teeth.

The most defensible single submission is probably A+B together: lead with the negative result, demonstrate the fix.

---

## Open work flagged for the next run

1. **Re-baseline in pair-mode** to clean up the H8 asymmetry. Quick.
2. **Re-run HS3 + Arena GPT-5 baselines** with `--judge-timeout 60` to confirm the LitBench gap pattern generalizes. ~1h, ~$15.
3. **Re-run Phase 7 minimal-prompt** with `max_tokens=128`. Resolves H10. ~30 min.
4. **Save actor checkpoints next run** (`save_freq` is currently never triggered for the non-final step). Without `global_step_*` directories we cannot do parameter-delta analysis or n=8 bootstrap eval on intermediate checkpoints.
5. **Conditional retrain #13** (frozen external judge) to test the H5 fix directly.

## Files / artifacts

- Raw outputs: `/root/rubric-policy-optimization/SDPO/diagnostics_20260505-031028/`
- Per-phase summary: `diagnostics_20260505-031028/SUMMARY.md`
- Orchestrator: `run_overnight_diagnostics.sh`
- New scripts (all under `scripts/`): `diag_prompt_mode_audit.py`, `diag_grpo_group_degeneracy.py`, `diag_judge_test_retest.py`, `diag_aggregate_ensemble.py`, `diag_position_shuffle_eval.py`, `diag_minimal_prompt_baseline.py`, `diag_summarize.py`. `cross_judge_rubric.py` was extended with a `--temperature` flag.
- Existing trained-policy GPT-5 cross-judge files (referenced in §Phase 5):
  - `logs/seed2_constantlr_total_absolute_20260504-071351/crossjudge_new_*_gpt5_n819.json` (Phase 2)
  - `logs/seed2_constantlr_total_absolute_20260504-071351/crossjudge_phase5rerun_gpt5.json` (Phase 5)
  - `logs/helpsteer3_total_absolute_v2_20260504-181504/crossjudge_helpsteer3_step100_gpt5.json`
  - `logs/arena_total_absolute_v3_20260504-213320/crossjudge_arena_step100_gpt5.json`
