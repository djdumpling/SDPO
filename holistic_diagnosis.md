# SDPO RuPO — Holistic diagnosis

State as of 2026-05-06 ~00:30 UTC. This document synthesizes all results from `observations.md`, `observations_v2.md`, `observations_v3_overnight_diagnostics.md`, last night's diagnostic battery (`diagnostics_20260505-031028/`), and tonight's A2 cross-judge confirmation on HelpSteer3 + Arena. It is intentionally blunt about what worked, what didn't, and where the project actually stands.

---

## Significance of the latest A2 results

The single most important thing about the A2 numbers is what they remove — they remove the last available "this might be a one-dataset fluke" defense. Until tonight, the headline finding (trained policy strictly worse than untrained baseline under GPT-5) only held on LitBench. Now it holds on all three preference benchmarks the project trained against.

| Dataset | Trained (GPT-5) | Untrained baseline (GPT-5) | Δ favoring baseline |
|---|---:|---:|---:|
| LitBench | 0.6640 (n=759) | **0.7651** (n=396) | **+10.1 pp** |
| HelpSteer3 | 0.7158 (n=438) | **0.7746** (n=284) | **+5.9 pp** |
| Arena | 0.6074 (n=940) | **0.6344** (n=279) | **+2.7 pp** |

The magnitudes are also load-bearing. The gap is largest on LitBench (creative writing, where Qwen3-8B has the strongest narrow priors), smaller on HelpSteer3 (general helpfulness), smallest on Arena (very diverse mix). That ordering is exactly what the cross-family-Goodhart hypothesis predicts: RL exploits Qwen3-8B's idiosyncratic preferences that don't generalize. On Arena there isn't a single pretrained-prior wall to climb, so the policy can't drift very far from baseline. On LitBench the wall is tall, the policy climbs it, and crashes.

You can no longer claim "the in-house judge isn't measuring the right thing but the trained model is fine." Both held-out judges agree, across three datasets, that the trained model is **worse**.

---

## Holistic synthesis — what was tried, why, what we learned

### Phase 1 — the original recipe (May 3)

**What was tried.** Five reward shapes (`margin`, `criteria_margin`, two `criteria_margin_bounded` variants, `criteria_total_absolute`), all on Qwen3-8B policy + frozen Qwen3-8B judge, GRPO with n=4 rollouts, LR=2e-6 cosine, 200 steps, single seed. SDPO loss layer with α=0.5 (orthogonal to GRPO, doing the JSD self-distillation).

**Why.** The hypothesis was that rubric *content* could be RL-shaped to improve preference accuracy on creative writing. Reward-shape ablation was the obvious first axis.

**What it found.** `criteria_total_absolute` peaked at val/reward 0.700 vs 0.60–0.68 for the other shapes. Picked as the working recipe.

**Real read in retrospect.** The "winner" was decided by a single self-judge measurement on a single seed under a noisy judge. That number was load-bearing for everything that followed, and we never validated that ranking under an external judge. In hindsight you should not trust a 7-pp ablation gap that's measured by a stochastic same-family judge.

### Phase 2 — constant LR fix (May 4)

**What was tried.** Same shape, constant LR instead of cosine, seed=1. 100 steps.

**What it found.** 0.6917 final — the strongest single number ever achieved. This became the headline result (5uwbvhqf).

**Real read.** This number turned out to be (a) within the noise floor of a single self-judge eval and (b) measured against a judge that systematically rates trained Qwen rubrics higher than they "deserve" by 8–10 pp under GPT-5. The +10 pp lift over cosine (0.5958 → 0.6917) is now suspect — possibly the constant-LR run just stayed in the regime where the same-family Goodhart loop was most exploitable.

### Phase 3 — multi-seed (May 4 overnight)

**What was tried.** Same recipe at seeds 2 and 3.

**What it found.** Seed 1: 0.6917. Seed 2: 0.6258 (-7 pp). Seed 3: 0.3779 (val pathology, not actually a -30 pp run but a structural rubric-format failure).

**Real read.** Across-seed variance was already a screaming red flag. A working method does not deliver 0.69 at seed 1 and 0.63 at seed 2 — that's a 14 pp seed-to-seed swing on n=819, far outside any reasonable bootstrap CI. The "reward shape matters" claim was effectively standing on a single seed.

### Phase 4 — cross-judge analysis (May 4)

**What was tried.** GPT-5 and Sonnet 4.6 re-judged val outputs to detect judge-specific gaming.

**What it found.** Two warning signs that were not adequately weighted at the time:

- **5uwbvhqf showed +13 pp self-vs-GPT-5 inflation** on the wandb subset (n=80). Sonnet didn't see it, so the team flagged it as "one-of-two suspicious" and moved on. In hindsight this was the H5 signal showing up.
- **Phase 5 (`criteria_margin` shape) showed −8.5 pp self-vs-cross gap** in the *opposite direction* — held-out judges rated `criteria_margin` rubrics higher than the in-house judge did. This was framed at the time as "rubric portability is real" — but the more likely explanation was that the in-house judge has a different idiosyncratic gradient than GPT-5 does, and whichever direction the policy drifts, you can find a held-out judge that disagrees. The "portability" framing was probably the wrong sign of the same underlying noise.

**Real read.** This was the moment where the project should have stopped and asked "are we measuring what we think we're measuring?" Instead it became "judge-portability is a feature."

### Phase 5 — three-dataset generalization (May 4–5)

**What was tried.** Apply the recipe to HelpSteer3 (helpfulness) and Arena (general chat). Train, cross-judge.

**What it found.** Self-judge: trained ≈ baseline on all three datasets (within ±1 pp). The "RuPO improves preference accuracy" claim could no longer be made even on the in-house metric — the framing pivoted to "RuPO steers rubric content."

**Real read.** This is where the project's epistemic state actually became "we have a method that does not measurably improve the metric we set out to improve, but it changes content in interesting ways." Defensible as an interpretability paper, but no longer the original NeurIPS pitch.

### Phase 6 — overnight diagnostic battery (May 5)

**What was tried.** Ten phases against twelve hypotheses for *why* the result was flat.

**What it found** (verdicts after cleanup):

| H | Hypothesis | Verdict | Force of evidence |
|---|---|---|---|
| H1 | Judge stochasticity swamps signal | confirmed at criterion level (SNR=0.52), but ensembling lift only +0.3–0.9 pp | Strong mechanism, weak fix |
| H2 | GRPO degeneracy from sparse reward | strongly confirmed: 65.6% of n=4 groups have zero advantage | Strong |
| H3 | Self-judge ceiling caps everyone at 0.69 | **decisively ruled out** — GPT-5 baseline on LitBench = 0.7651 | Decisive |
| H5 | Cross-family Goodhart | strongly confirmed on LitBench: trained 0.66 vs baseline 0.77 under GPT-5 | Decisive |
| H7 | Position bias | ruled out (flipped acc = 0.307 ≈ 1−0.69) | Decisive |
| H8 | Train/val prompt-mode mismatch | hallucinated by my summary template — both are pair-mode | Decisive (negative) |
| H10 | Prompt template carries all the lift | unresolved (Phase 7 parser bug) | Pending |

**Real read.** The diagnostic battery did its job — it identified the actual mechanism with quantitative evidence. The reward-shape ablation, the constant-LR fix, the multi-seed runs, all of them were measuring noise on top of a structurally broken setup.

### Phase 7 — A2 confirmation (May 6 early UTC)

**What was tried.** Repeat the GPT-5 baseline cross-judge on HelpSteer3 and Arena (last night these stalled on Pinference).

**What it found.** All three datasets confirm the H5 direction. Trained policies are strictly worse than the untrained baseline under GPT-5, by 2.7–10.1 pp depending on dataset.

---

## What actually worked vs what didn't

### Worked (and is keepable)

1. **The methodology stack.** Paired bootstrap CIs, matched-row cross-judge comparisons, n-asymmetry awareness, the full diagnostic-battery scaffolding. This is reusable infrastructure that any future rubric-RL project should adopt — the cross-judge n=192 sampling artifact caught earlier is exactly the sort of mistake the field keeps making.
2. **The cross-judge framework as a falsification tool.** It found the central problem in this project. It will find similar problems in future ones.
3. **Rubric content analysis.** The vocab-collapse finding (`emotional X`, `clarity X` dominating after training) is real and quantitatively documented across all three datasets. It's a genuine interpretability artifact even if it doesn't help the headline claim.

### Didn't work

1. **The core RuPO recipe** as built — Qwen3-8B policy + Qwen3-8B judge + GRPO + n=4 + sparse ternary reward — is structurally broken. Three independent failure modes compound: GRPO is starved of gradient on 65% of batches, the judge has SNR<1 on the remaining batches, and the surviving signal is a same-family Goodhart loop.
2. **Reward-shape variation.** None of the shapes produces trained > baseline under any external judge. The "criteria_total_absolute is best" finding was an artifact of measuring against the same family's noise.
3. **LR schedule variation.** Cosine vs constant moves the in-house number around but doesn't change the cross-judge verdict.
4. **Multi-seed.** Two clean seeds, 7 pp apart on self-judge, same-or-worse than baseline on GPT-5. There is no robust signal here.

---

## Why this is not NeurIPS-submittable

1. **The headline claim is falsified, not weakly supported.** "RL on rubric content improves preference accuracy" is false on every external metric we measured, on every dataset, by margins large enough that variance reduction (n=8 rollouts, k=4 ensembling) won't paper over them.
2. **The interpretability framing is a *negative* contribution.** "Look how training warps rubric content" is honest, but reviewers will read it as "RL broke your model and you're trying to spin it." Vocab collapse is an interesting finding, but the right venue for that is an interpretability workshop, not a main-conference method paper.
3. **The conditional fix (cross-family judge retrain) hasn't run** and even if it works, it's a different method. SDPO's whole identity is "self-distillation" — once policy ≠ judge, you don't have RuPO-as-conceived, you have a more conventional RLHF-with-external-judge setup. The contribution shrinks.
4. **The variance-reduction story is dead on arrival.** Even with judge SNR fixed (k=4 ensembling), the lift is 0.3–0.9 pp. This won't unlock a positive result; it will, at best, make the negative result statistically tighter.
5. **The reward-shape ablation is now suspect.** That was meant to be one of the empirical contributions. With the in-house judge revealed as miscalibrated against external judges, the ranking of shapes by self-judge val/reward is not trustworthy.

What remains is a strong, well-documented, reproducible **negative result with a clean mechanistic explanation**. That's a real thing. It is not a NeurIPS main-conference paper.

---

## Where to go from here

In rough order of effort vs payoff. Honest about what each one actually buys.

### 1. Workshop / position paper: "Diagnosing Goodhart in self-judge rubric RL"

**What it would say.** Train a policy with a same-family judge, you get this characteristic failure: in-house metric flat, cross-family judge agreement strictly negative, GRPO degeneracy >50%, vocab collapse onto judge-favored bigrams. Here's a four-test diagnostic battery (cross-judge gap, group-degeneracy fraction, judge SNR, parameter delta). Adopt this before claiming improvement.

**Effort.** Low — most of the writing is already done across the three observations docs. Maybe 2 weeks. **Payoff.** Modest. Workshops at NeurIPS/ICLR. Useful methodology contribution. Doesn't move the research-direction needle much.

### 2. Cross-family-judge retrain (B0/B1, currently blocked on the wedged CUDA driver)

**What it would test.** Whether swapping in GPT-5 (or Claude, or RubricRM-4B) as training-time judge breaks the Goodhart loop and produces trained > baseline under GPT-5 eval.

**Effort.** ~$300 + a few hours of work, blocked right now on the host's wedged CUDA driver. Launchers (`run_retrain_gpt5_judge.sh`, `run_next_experiments_20260505.sh`) are built and ready to fire as soon as CUDA recovers. **Payoff.** If it works: a positive method paper, but it's "RuPO with cross-family judge" — meaningfully different from the original pitch and arguably less novel (replacing self-distillation with off-the-shelf RLHF). If it doesn't work: the method is dead independent of the judge family, and you've spent $300 to confirm you should pivot.

**Honest read.** Run this. It's the only experiment that can convert the negative result into something positive without abandoning the project entirely. Plan for it to fail — set up the pivot path before kicking off the spend.

### 3. Verifiable-reward pivot

**What it would test.** Replace the LLM judge with a deterministic scorer — e.g., rubric coverage of human-annotated preference dimensions, or a learned classifier on human pairwise preferences (a Bradley-Terry RM trained on the same data). Sidesteps Goodhart entirely because the reward is no longer "what does another LLM think."

**Effort.** Medium. Need to define the verifiable reward and validate it tracks human agreement. ~3–4 weeks of work. **Payoff.** Higher upside than (2) — moves the field forward by removing the LLM-judge dependency. Less novel as a method, more solid as engineering.

### 4. Interpretability paper on vocab collapse

**What it would say.** Quantify how RL on a stochastic same-family judge concentrates rubric-criterion vocabulary onto a few attractors (`emotional X` on creative writing, `clarity X` on chat). Connect this to the Goodhart analysis. Probe what the policy gradient is actually selecting for.

**Effort.** Low–medium. Most of the data exists. **Payoff.** Workshop-paper level. Honest, clean, but not a career-making contribution.

### 5. Pivot away from RuPO entirely

The brutal-honesty option. The diagnostic stack and the methodology lessons transfer to any rubric-RL project. The specific recipe doesn't. There may be more leverage in starting fresh with a different framing — e.g., **"rubrics as reasoning traces"** (use rubrics as *intermediate reasoning* for the judge, not as a reward target) or **"rubric-conditioned generation"** (generate responses conditioned on rubrics, evaluate the responses, not the rubrics) — than in trying to rescue this one.

---

## Recommendation

If there is $300 and 12 hours of GPU time before any deadline, run B0/B1 with GPT-5-as-training-judge. Cheapest path to either a positive headline or a definitive "the entire framework is dead" verdict. Don't conflate "want this to work" with "this might work" — it might not.

If B0/B1 fails, the most defensible move is to combine **(1) workshop paper on the diagnostic battery + (4) interpretability paper on vocab collapse**, then pivot the research direction to (3) verifiable rewards or away from rubric-RL entirely. Total time to get those two writeups out: ~4–6 weeks. They won't go to NeurIPS but they're real publications and they preserve the engineering work.

Do *not* try to rescue the original framing. The result is consistent across three datasets, two reward families (margin and absolute), two LR schedules, two seeds, and two held-out judges. That's not a marginal failure that variance reduction can fix; it's a structural one.

---

## File pointers

- Earlier observations: `observations.md`, `observations_v2.md`, `observations_v3_overnight_diagnostics.md`
- Diagnostic battery raw outputs: `diagnostics_20260505-031028/`
- A2 outputs (HS3 + Arena GPT-5 baselines): `next_experiments_20260505-232440_a2only/pA2_gpt5_baselines/`
- Retrain launcher (ready to run): `run_retrain_gpt5_judge.sh`
- Orchestrator with pilot phase: `run_next_experiments_20260505.sh`
- Existing trained-policy GPT-5 cross-judge files:
  - `logs/seed2_constantlr_total_absolute_20260504-071351/crossjudge_new_*_gpt5_n819.json` (Phase 2 LitBench)
  - `logs/seed2_constantlr_total_absolute_20260504-071351/crossjudge_phase5rerun_gpt5.json` (Phase 5 criteria_margin)
  - `logs/helpsteer3_total_absolute_v2_20260504-181504/crossjudge_helpsteer3_step100_gpt5.json`
  - `logs/arena_total_absolute_v3_20260504-213320/crossjudge_arena_step100_gpt5.json`
