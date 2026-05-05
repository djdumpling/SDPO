# Diagnosing why RuPO ≈ untrained baseline on preference accuracy

*Plan revision — 2026-05-04. Supersedes the previous plan in `lovely-hatching-axolotl.md`. Incorporates OpenRubrics/RubricRM methodology divergence flagged by Anthony.*

## Context

The submission was supposed to read "RuPO improves preference accuracy via RL on rubric content." The data instead says: across **three datasets** (LitBench / HelpSteer3 / Arena), trained Qwen3-8B and untrained Qwen3-8B both score ≈0.69 / ≈0.69 / ≈0.58 on self-judge preference accuracy. Training delta is essentially zero or slightly negative even at peak step. This contradicts the central premise.

**Goal of this plan: diagnostically isolate which 1–2 of {training-method pathology, judge/eval setup, prompt-template ceiling, dataset/judge accuracy ceiling, model size} explain the parity**, then either fix the broken thing or refrane the paper around the real finding. Most diagnostics are read-only (compute on existing logs); the few intervention experiments are gated on what those analyses find.

---

## What we already know

### Our training setup (from code audit)

- **Reward shape `criteria_total_absolute` is ternary {0, 0.5, 1.0}** — `SDPO/environments/dpo_to_rupo_verl/reward_fn.py:178-179`. With n=4 GRPO rollouts per prompt and `norm_adv_by_std_in_grpo: False` (`SDPO/verl/trainer/config/sdpo_rupo.yaml:67`), most groups will be all-1 or all-0, producing zero advantage.
- **Same Qwen3-8B serves as policy AND judge**, both at temperature 0.7. The judge is re-sampled stochastically every step; per-criterion score variance is plausibly ±5–15 on a 0–100 scale. **Single judge call per criterion per response — no averaging.**
- **No KL constraint, no entropy bonus** (`use_kl_loss: False`, `use_kl_in_reward: False`, `actor.entropy_coeff: 0`).
- **LR=2e-6, constant, 100 steps × 72 prompts × 4 rollouts = 7,200 prompts seen (~0.48 epochs)**.
- **Validation uses n=1 rollout at temp=0.7** (`VAL_KWARGS_N=1`). The 0.69 numbers are themselves single-sample noisy estimates.
- **Max tokens: prompt 4096 / response 4096** (very loose; rubric outputs can ramble to 1.6k+ chars).

### Empirical signals from existing logs

- Per-step rollout reward oscillates 0.34–0.90 around mean 0.615 (which is *below* the val baseline 0.69) — train-eval mismatch suggests train signal is noise-dominated.
- Chosen−rejected score-gap variance is huge: range −14.86 to +30.11, mean 9.35. **Mean signal ≈ 1σ of judge noise.**
- Rubric parse rate = 100% in trained — format constraints are not the gradient sink.
- Trained rubrics ARE substantively different from baseline (longer +18%, "emotional X"-heavy on LitBench; "clarity X" on Arena/HS3) and ARE domain-specific. **Training is changing rubric content; it's just not changing accuracy.**
- Cross-judge gap on Phase 5 (`criteria_margin`) is significantly negative — held-out judges rate trained rubrics +9–11pp higher than the in-house judge.

### Methodology divergence from published rubric-RL work

This is the most important new context. Three recent papers all use radically different setups than ours:

| Setting | Ours (RuPO/SDPO) | OpenRubrics / RubricRM ([2510.07743](https://arxiv.org/abs/2510.07743)) | RaR ([2507.17746](https://arxiv.org/abs/2507.17746)) |
|---|---|---|---|
| **Rubric generator temp** | 0.7 | **0 (deterministic)** | 0 |
| **Judge temp** | 0.7 (single call) | **0.7–1.0, sampled multiple times** | sampled, averaged |
| **Rollouts per prompt (n)** | **4** | **64 / 128 / 244** (depending on stage) | k≥8 |
| **Reward shape** | ternary {0, 0.5, 1.0} | dense per-criterion scoring averaged across judge samples | dense Likert averaged |
| **Rubric source** | policy generates fresh per prompt | **Contrastive Rubric Generation** — derives rubrics by contrasting chosen vs rejected; produces hard rules + principles | curated by GPT-4 |
| **Max tokens (rubric)** | **4096** (`run_sdpo_qwen35_9b_litbench_8gpu.sh:61`, `MAX_RESPONSE_LENGTH`) | **1024** (forces concision) | not specified |
| **Max tokens (judge)** | 4096 (`JUDGE_MAX_TOKENS`, line 150) | 4096 | 4096 |
| **Policy/judge weight coupling** | **strict identity** — judge served from the same vLLM instance as the policy (`JUDGE_MODEL=$MODEL_PATH`, line 147; `OPENAI_BASE_URL=$JUDGE_BASE_URL` points at local serving). Judge weights drift with policy weights at every gradient step | **decoupled** — Rubric-RM-4B is trained once offline and frozen during downstream policy training. Policy never produces rubrics; rubrics are baked into the RM | **decoupled** — judge is a fixed external model (GPT-4 / Claude) |
| **Headline result** | trained ≈ baseline | RubricRM +8.4% over size-matched baselines; policies trained with it +3-7pp on IFEval/InfoBench | RaR +31% on HealthBench |

The pattern is unmistakable: **everyone who beats the baseline in this space (a) reduces judge variance via ensembling, (b) uses many more rollouts per prompt, and (c) does not couple judge weights to policy weights**. We're doing the opposite on every axis.

### Token-budget asymmetry (specifically)

Rubric is 4× longer than OpenRubrics allows. The +18% rubric-length drift between trained vs untrained (1649 vs 1398 chars) suggests training is exploiting the loose cap — emitting verbose rubrics that the judge then has to apply to long responses. Long rubrics × long responses = more variance per criterion score per call (more tokens for the judge to attend over inconsistently). OpenRubrics' 1024 cap is plausibly part of why their judge calls are less noisy.

### SDPO vs GRPO — what each name actually controls in this codebase

The `sdpo_rupo.yaml` config sets two **orthogonal** layers (sources of common confusion):

- **`adv_estimator: grpo`** (line 67) — *how advantages are computed.* Group-relative, no critic, advantage = (reward − group-mean) / group-std (or just group-mean here, because `norm_adv_by_std_in_grpo: False`).
- **`loss_mode: sdpo`** (line 33) — *the actual loss objective.* This codebase's "SDPO" is a verl extension that adds:
  - JSD-based loss with `alpha: 0.5` (mixes mode-seeking forward-KL with mode-covering reverse-KL — line 48)
  - **Self-distillation reprompts**: when a rubric fails, the policy is re-prompted with the failed rubric + the judge's feedback + a successful rubric as in-context demonstration (lines 34-57). This is the "self-distillation" core — the policy learns from its own successful rollouts as exemplars during training.

So the codebase's "SDPO" = **GRPO advantages + JSD loss + on-policy self-distillation reprompts.** It is NOT the offline-DPO algorithm of the same name in the broader literature; it does use rollouts and a reward function. **All of the GRPO pathologies below (degenerate groups, std-norm off, n=4 too small, judge-policy weight coupling) apply unchanged — the SDPO loss layer reshapes the per-token gradient *after* advantages are computed; it does nothing to fix upstream reward signal collapse.**

### Critical literature on the ceiling

- **LitBench paper** ([2507.00769](https://arxiv.org/abs/2507.00769)): best zero-shot LLM judge (Claude-3.7-Sonnet) reaches **73%** human agreement. GPT-4.1 / Deepseek-R1: 70–71%. Fine-tuned (supervised) Bradley-Terry/Generative RMs: 78%. **Our 0.69 is right at the unsupervised-LLM-judge ceiling.**
- **GRPO degenerate-group failure mode** ([RC-GRPO 2602.03025](https://arxiv.org/abs/2602.03025), Reg-GRPO): peaked SFT initialization + sparse rewards → groups collapse to identical reward → zero advantage → no learning. Our setup matches this profile.

---

## Hypotheses (re-ranked given OpenRubrics divergence)

### Tier 1 — Most likely + highest prior given methodology gap

**H1 (was H3). Judge stochasticity at temp=0.7 with no averaging dominates the rubric signal.** [PROMOTED to top tier.] The chosen−rejected mean gap = 9.35 with σ ≈ 11+ across rollouts means SNR ≤ 1. OpenRubrics and RaR both ensemble the judge over k≥4 samples; we use k=1. With per-call judge noise σ ≈ 5-15pt per criterion, averaging k=4 calls cuts σ by 2× — directly recovering signal we're throwing away.
- *Falsifier:* re-judge a fixed set of ~100 (rubric, chosen, rejected) triples 5× each at temp=0.7. Compute per-triple test-retest σ vs between-rubric σ. If test-retest σ ≥ between-rubric σ, judge noise dominates. Then test: re-eval with k=4 averaged judge calls — accuracy delta vs k=1 is the noise-floor lift.
- *Cost:* ~$2 API + 30 min if local Qwen3-8B vLLM is up; ~30 min GPU.

**H2. GRPO group-advantage degeneracy from sparse ternary reward × n=4.** [Compounds with H1.] With reward ∈ {0, 0.5, 1.0} and n=4 rollouts, a large fraction of groups collapse to all-same → advantage=0. OpenRubrics uses n=64/128/244 — they have ~16-60× more samples per advantage estimate than us.
- *Falsifier:* Mine Phase 2 training stdout/wandb logs for per-step rollout-group reward distributions. Compute % of groups where all 4 rollouts received the same reward. If >40%, this is a major signal-killer; if combined with H1 (judge noise even when groups vary), much of the "non-degenerate" advantage is noise too.
- *Cost:* 30–60 min, free.

**H3. Self-judge ceiling: Qwen3-8B's judge agreement with humans tops out around 0.69–0.73 on LitBench-style data.** No matter what rubric the policy emits, judge-as-evaluator caps us. Untrained ≈ trained ≈ ~0.69 ≈ judge ceiling.
- *Falsifier:* re-evaluate ONE trained checkpoint AND base baseline using **GPT-5** as the judge with the SAME rubrics. If both jump to 0.72–0.78 *together with the same delta*, judge ceiling is binding. If GPT-5 reveals a gap (trained > base by 2pp+), then training IS extracting signal — Qwen-judge just can't see it.
- *Cost:* ~$30–60 API for n=819 × 2 runs × 1 dataset.

**H4. Eval is itself a noisy single-sample estimate.** VAL_KWARGS_N=1 at temp=0.7. The 0.69 vs 0.69 "parity" might just be inside the noise floor of single-rollout val.
- *Falsifier:* Re-eval one trained checkpoint and one baseline with n=8 rollouts × temp=0.7 on same val set; bootstrap accuracy CI. If 95% CI is wider than ±0.02, "parity" is stochastic indistinguishability, not ceiling.
- *Cost:* 30 min GPU, free.

### Tier 2 — Plausible, moderate cost

**H5. Moving-target judge: judge weights are LITERALLY the policy weights at the current step.** Stronger than typical Goodhart: not "the policy exploits a fixed judge's biases," but "the policy is optimizing against a target that drifts with it every gradient step, like a GAN where generator and discriminator share weights." The judge served from the local vLLM is `JUDGE_MODEL=$MODEL_PATH` — the same checkpoint being trained. As the policy shifts to favor "emotional X" criteria, the judge *also* shifts to score those criteria more confidently, validating the policy's drift without external grounding.
- *Already-supporting evidence:* Phase 5's significant negative cross-judge gap (held-out judges DON'T agree with the trained-judge's increased confidence). Vocabulary collapse to "emotional X." Train-side reward 0.615 < val-side reward 0.69 — train and val see different judge weights at any given step (the val judge is a snapshot from the eval call).
- *Falsifier:* Train one short run (25 steps) using a *frozen* held-out judge (e.g., RubricRM-4B served locally, OR a frozen Qwen3-8B at step-0 checkpoint pinned on a separate vLLM port) for the *training-time* reward. If self-judge eval improves *and* cross-judge doesn't show negative shift, the moving-target effect was the dominant pathology.
- *Cost:* ~3h GPU + ~30 min for separate judge serving setup.

**H6. Insufficient training (LR + steps + KL-free).** LR=2e-6 over 100 steps with no exploration bonus and no KL anchor → policy doesn't move enough. Compounds with H1+H2.
- *Falsifier:* Measure mean |Δθ| between step 0 and step 100 actor params. If <1% of base parameter scale, the policy is effectively unchanged.
- *Cost:* parameter-delta check is free (5 min).

**H7. Pair-mode prompt leakage / position bias.** Training prompt shows the policy chosen+rejected. Despite anti-leak instructions, the policy may learn position priors that don't transfer to flipped-position eval.
- *Falsifier:* Re-eval with chosen/rejected swapped. If accuracy drops <0.5, strong position bias.
- *Cost:* 30 min, free.

**H8. Train uses pair-mode but val (and untrained baseline) uses prompt-only mode.** Could fully invalidate the comparison.
- *Falsifier:* Audit `policy_prompt_mode` for both train and val in the trainer config. **Run this first** — precondition for all other diagnostics.
- *Cost:* 10 min, free.

**H9. Policy temperature 0.7 at evaluation introduces format / rubric instability that the deterministic baseline avoids.** OpenRubrics uses **rubric-temp=0** at evaluation. Our policy emits a different rubric every val rollout because temp=0.7. Variance in the *rubric itself* (separate from judge variance) likely accounts for some of the noise floor.
- *Falsifier:* Re-eval one trained checkpoint and one baseline with **policy temp=0** (deterministic rubric) and same temp=0.7 judge. If accuracy gap or CI shrinks, policy stochasticity is contributing to the parity.
- *Cost:* 30 min GPU, free.

### Tier 3 — Plausible but expensive / hard to falsify in budget

**H10. Dataset label-noise ceiling.** LitBench/HS3/Arena pairs are noisy enough that the Bayes-optimal classifier under any judge ≈ 0.70–0.75.
- *Falsifier:* Filter to "high-confidence" subset (HS3 `overall_preference ∈ {±3}`, LitBench annotator-agreement-tagged splits). Re-eval base + trained on easy subset. If both jump together to 0.85+, we're at noise floor on full set; if both stay flat, label noise isn't binding.
- *Cost:* free re-eval on filtered subset.

**H11. Prompt-template effect (instruction tuning + structured XML format)** does most of the lift; RL has nowhere to go.
- *Falsifier:* Run minimal-prompt baseline ("which response is better, A or B?") with Qwen3-8B on LitBench val. If ≥0.65, the rubric template wasn't doing much; if drops to ~0.55, template carried the signal.
- *Cost:* 30 min GPU, free.

**H12. Model-size effect.** 8B is too small for RL to find new modes beyond what instruction tuning encoded.
- *Falsifier:* Train Qwen3-4B same recipe; compare delta-from-baseline.
- *Cost:* ~3h GPU. Gated on Tier 1 ruling out other causes.

**H13. Rubric max-tokens too generous (ours: 4096 vs OpenRubrics: 1024).** Confirmed numbers — `MAX_RESPONSE_LENGTH=4096` is the rubric cap (the policy can ramble for 4096 tokens), while OpenRubrics uses 1024. Long rubrics × long responses = more tokens for the judge to attend over inconsistently per criterion → more variance. The +18% rubric-length drift in trained vs untrained suggests training is exploiting this slack. Judge max_tokens (4096) matches OpenRubrics — that knob is fine.
- *Falsifier:* Re-generate baseline rubrics with `max_tokens=1024`; check if length truncation drops accuracy (cap is biting) or doesn't (cap is slack and not the cause). Then re-eval trained rubrics with judge applying truncated-to-1024 versions.
- *Cost:* 30 min GPU. Defer until Tier 1 results.

---

## Diagnostic queue (cheapest, highest-information first)

| # | Test | Hypothesis | Cost | Decision triggered |
|---|---|---|---|---|
| 1 | **Audit train vs eval prompt mode** in `run_sdpo_qwen35_9b_litbench_8gpu.sh` + `preprocess_dataset.py` | H8 | 10 min, free | If mode differs → fix and re-eval baseline; could resolve everything |
| 2 | **Mine Phase 2 logs for GRPO group-reward distribution** (% degenerate groups; advantage-zero fraction) | H2 | 30 min, free | If >40% degenerate → top training-method pathology |
| 3 | **Judge test-retest** on 100 fixed (rubric, response) pairs × 5 samples each at temp=0.7 — quantify per-call σ | H1 | 30 min, ~$2 (or free local) | Quantifies judge SNR. Decision-defining for the whole investigation |
| 4 | **Bootstrap val CI** with n=8 rollouts (policy temp=0.7) on one trained + one baseline | H4 | 30 min GPU | Establishes whether "parity" is noise or signal |
| 5 | **Deterministic policy eval** (policy temp=0, judge temp=0.7 single call) | H9 | 30 min GPU | Isolates policy stochasticity from judge stochasticity |
| 6 | **Judge ensembling re-eval** — same rubrics, judge sampled k=4 times averaged | H1 | 30 min GPU + small API | Is the OpenRubrics-style fix |
| 7 | **Position-shuffle val** (swap chosen/rejected) | H7 | 30 min, free | Tests for position bias / pair-mode leakage |
| 8 | **High-confidence subset re-eval** on HS3 (strong-preference rows) | H10 | 30 min, free | Tests label-noise ceiling |
| 9 | **Minimal-prompt baseline** Qwen3-8B on LitBench val | H11 | 30 min GPU | Estimates how much of 0.69 is from prompt template |
| 10 | **GPT-5 judge re-eval** on base + trained rubrics, LitBench n=full | H3 | 1h, ~$30–60 | Most decisive single test for self-judge ceiling vs training-method failure |
| 11 | **Parameter-delta check** between step 0 and step 100 actor weights | H6 | 5 min, free | Quantifies whether the policy actually moved |

**Steps 1–11 ≈ 4h elapsed + ~$70 API.** They produce a defensible diagnosis.

### Note on rollout-budget tradeoff

n=4 rollouts/prompt × ternary reward gives degenerate GRPO advantages. The naive fix is "use n=64 like OpenRubrics" but each rollout costs ~10 judge calls (1 chosen + 1 rejected, scored on each of ~5 criteria). Going from n=4 → n=64 multiplies judge cost by 16×.

Better resource allocation: split the variance-reduction budget between *policy diversity* (rollouts) and *judge stability* (per-call ensembling). Judge noise is on a 0-100 scale per criterion while reward is on {0, 0.5, 1.0} — every dollar invested in judge stability tightens the reward signal more than the same dollar spent on more rollouts. **Recommended: n=8 with judge ensembled k=2-4 averaged. That's 4× current judge calls, not 16×.** The averaged judge score also makes the reward effectively continuous (no longer ternary), which is the deeper fix for H2.

### Conditional intervention experiments (gated on diagnosis)

| # | Test | Hypothesis | Cost | Trigger |
|---|---|---|---|---|
| 12 | **Retrain with OpenRubrics-aligned recipe**: rollout policy temp=0.7 (need diversity for GRPO advantages) BUT eval policy temp=0; **judge temp=0.7 ensembled k=2-4 averaged**; **n=8 rollouts/prompt** (NOT n=16 — pair with judge ensembling for better SNR per dollar); std-normalized GRPO (`norm_adv_by_std_in_grpo: True`); LR=1e-5 (5× current); rubric max_tokens=1024; same `criteria_total_absolute` reward but compute on the *averaged* judge score so it's no longer ternary | H1+H2+H6+H9+H13 | ~8-10h GPU + $20-40 API (k=2 doubles judge cost; n=8 doubles rollout cost; net ~4× current judge calls) | Run if Steps 1-6 confirm noise-domination as the binding constraint |
| 12b | **Cheap variance-reduction-only ablation**: keep n=4 rollouts but enable std-norm GRPO + judge ensemble k=4 + LR=1e-5 | H1+H2+H6 | ~5h GPU + small API | Run first as a cheaper version of #12 if budget is tight; isolates "did variance reduction alone fix it" from "did we also need bigger n" |
| 13 | **Train with frozen held-out judge** — pin a *separate* vLLM port serving step-0 Qwen3-8B, point training judge calls there; the policy still trains but the judge weights are now frozen | H5 (moving target) | ~3h GPU + ~30 min serving setup | Run if Step 10 shows GPT-5 reveals a real trained>base gap on existing checkpoints (i.e., training IS extracting signal but in-house judge can't see it) |
| 14 | **Qwen3-4B same recipe** (sanity check on model size) | H12 | ~2h GPU | Run only if Tier 1 hypotheses don't explain the result |

---

## Decision tree / expected outcomes

After steps 1–11, the result lands in one of three scenarios:

**Scenario A — "Noise floor + judge-method mismatch" (most likely).** Step 3 shows judge test-retest σ ≥ between-rubric σ → judge noise dominates. Step 6 (k=4 ensembling) shows accuracy lift of 1-3pp, AND Step 4 reveals val CI of ±0.025+. Step 2 shows GRPO group-degeneracy >30%. Step 10 shows GPT-5 judge gives slightly higher and tighter accuracy. → **Path: Run experiment #12 (OpenRubrics-aligned retrain). Paper claim becomes "RuPO improves accuracy when implemented with proper variance reduction (ensembled judge + n=8 rollouts + deterministic eval), validated on three datasets."**

**Scenario B — "Self-judge ceiling + LitBench label noise."** Step 3 shows reasonable judge SNR. Step 6 ensembling barely moves anything. Step 10 GPT-5 judge bumps both trained and base to 0.72-0.78 *with the same delta*. → **Path: Don't retrain; reframe paper. Headline becomes "RuPO operates at the LLM-judge ceiling on standard preference benchmarks; gains are not measurable via self-judge eval, but are measurable via cross-judge shift and rubric content steering" — with quantitative ceiling argument from LitBench paper itself.**

**Scenario C — "Eval setup is broken."** Step 1 reveals mode mismatch, step 7 reveals position bias, or step 11 reveals near-zero parameter delta. → Bug-fix, re-run, story fundamentally changes.

**Most likely realistic outcome:** A + B mixture — judge ceiling is real (0.73 max) so we're *near* but not *at* the ceiling, AND noise floor is wasting most of the headroom. Running #12 (retrain with OpenRubrics methodology) tests whether we can reclaim the gap.

---

## Critical files / line refs

- `SDPO/run_sdpo_qwen35_9b_litbench_8gpu.sh:32, 41, 42, 81-86` — reward type, ROLLOUT_N=4, LR=2e-6, VAL_KWARGS_N=1
- `SDPO/verl/trainer/config/sdpo_rupo.yaml:62-67` — GRPO algo, `norm_adv_by_std_in_grpo: False`, rollout temp 0.7
- `SDPO/environments/dpo_to_rupo_verl/reward_fn.py:155-179` — `_clipped_margin_reward` and `_absolute_reward` formulas
- `SDPO/environments/dpo_to_rupo_verl/reward_fn.py:512-534` — judge-call structure (per-criterion, chosen+rejected separately, **single call no averaging**)
- `SDPO/environments/dpo_to_rupo_verl/preprocess_dataset.py:122-176` — `policy_prompt_mode` routing
- `environments/dpo_to_rupo/dpo_to_rupo/prompts.py:43-171` — pair vs prompt-only system prompts (audit for H8)
- `environments/dpo_to_rupo/dpo_to_rupo/judge.py:290-320` — judge prompt
- `SDPO/logs/seed2_constantlr_total_absolute_20260504-071351/` — Phase 2 training logs (mine for H2)
- `SDPO/logs/untrained_baseline_*/val_generations/0.jsonl` — base-model rubrics (use for steps 3, 5, 6, 10)

---

## Verification

For each hypothesis, the falsifier must produce one of:
- **Confirmed:** measurable signal change ≥1.5σ above noise (e.g., judge-noise σ > rubric-effect σ for H1; group-degeneracy fraction > 40% for H2; ensembling accuracy lift ≥ 2pp for the H1 fix; GPT-5 accuracy bump for H3).
- **Ruled out:** null result with bootstrap CI tight enough to make the hypothesis quantitatively implausible.

Each step's result is appended to `diagnostics.md` with: hypothesis, test, raw data, conclusion, next-step gate.

Final deliverable: one paragraph naming the 1–2 binding constraints with quantitative evidence. That paragraph either goes into the paper as the methodological-limitations section (Scenario B), or motivates retrain #12 (Scenario A).

---

## Sources

- [LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing](https://arxiv.org/abs/2507.00769) — judge ceiling = 0.73
- [OpenRubrics: Towards Scalable Synthetic Rubric Generation](https://arxiv.org/abs/2510.07743) — Contrastive Rubric Generation, n=64-244, judge ensembling
- [Rubrics as Rewards (RaR)](https://arxiv.org/abs/2507.17746) — +31% on HealthBench with judge averaging
- [RC-GRPO: degenerate-group failure mode](https://arxiv.org/abs/2602.03025) — sparse-reward + peaked-policy advantage collapse
- [Chasing the Tail: Effective Rubric-based Reward Modeling](https://arxiv.org/abs/2509.21500) — high-reward tail sharpening
