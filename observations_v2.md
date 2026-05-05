# SDPO RuPO — Final consolidated observations (post multi-dataset battery)

State as of 2026-05-05 ~01:30 UTC. Full plan executed; all 3 datasets trained + cross-judged + untrained-baselined; rubric content analyzed.

## TL;DR — the story has changed substantially

The intended NeurIPS narrative was "RuPO produces good rubrics that generalize across domains." The data partially supports that, but with a **very important caveat**:

> **The base Qwen3-8B model already writes rubrics that achieve ~equal preference accuracy as the RL-trained policy on the in-house judge.** Across all three datasets, training delta is essentially zero or slightly negative.

What training DOES change is the *content* of the rubrics — and whether held-out judges agree with the in-house judge. The cross-judge gap analysis is now the most defensible novel contribution.

## 1. Trained vs untrained baselines — the new headline table

All numbers use the same recipe: Qwen3-8B + `criteria_total_absolute` + constant LR + seed=1 (or seed=2 for Phase 2), 100 training steps, in-house Qwen3-8B judge.

| Dataset | Trained PEAK (self-judge) | Trained FINAL | Untrained baseline | Δ (peak − untrained) |
|---|---:|---:|---:|---:|
| LitBench-HA (5uwbvhqf, seed=1) | 0.7027 @ step 75 | 0.6917 | 0.6947 | **+0.008** |
| LitBench-HA (Phase 2, seed=2) | 0.6819 @ step 50 | 0.6258 | 0.6947 | **−0.013** |
| HelpSteer3 | 0.693 @ step 25 | 0.602 | 0.696 | **−0.003** |
| Arena-140k subsample | 0.5749 @ step 25 | 0.5613 | 0.5809 | **−0.006** |

**Reading.** The base model's rubric-writing ability is the dominant signal. RL training adds at best ~+1 pp at peak (LitBench seed=1) and otherwise mildly hurts. Training past ~25 steps actively decreases preference accuracy on every dataset (overfitting / Goodhart on the in-house judge).

## 2. Cross-judge results (held-out evaluators)

GPT-5 (and Sonnet 4.6 where available) re-scored val outputs with paired bootstrap CIs (5000 resamples).

| Run | Self (paired) | GPT-5 cross | Paired self−cross gap | Sig? (95% CI) |
|---|---:|---:|---:|---|
| LitBench 5uwbvhqf | 0.7688 (n=80*) | 0.6375 | **+0.131** | ✓ [+0.025, +0.238] |
| LitBench Phase 2 (full) | 0.6686 (n=759) | 0.6640 | +0.005 | ns [−0.026, +0.036] |
| LitBench Phase 5 (`criteria_margin`) | 0.5966 (n=810) | 0.6815 | **−0.085** | ✓ [−0.110, −0.060] |
| HelpSteer3 | 0.6781 (n=438) | 0.7158 | −0.038 | marginal [−0.078, +0.002] |
| Arena | 0.5793 (n=940) | 0.6074 | **−0.028** | borderline [−0.056, +0.000] |

\* The 5uwbvhqf wandb-table subset is upward-biased — full-set self-judge is 0.6917, so the +0.131 gap is partly a sampling artifact. The Sonnet cross-judge (also n=80) sees gap = 0.000, suggesting the GPT-5 result reflects GPT-5's particular taste, not robust inflation.

**Reading.**
- **Phase 5 (criteria_margin) shows a strongly significant negative gap** — both held-out judges rate it ~9-11 pp higher than the in-house judge does. This is the cleanest "training shifts the policy in a direction Qwen-judge under-rates but external judges value" finding.
- **HS3 and Arena directionally line up with Phase 5** but at smaller magnitude.
- **No run shows reward-hacking signature** (positive gap with held-out judge agreement).

## 3. Rubric content analysis (interpretability, n=11 runs)

The full analyzer output is at `/tmp/rubric_analysis.txt`. Highlights:

### Validity rate (rubric XML parses + has ≥2 weighted criteria summing to 100)

| | step 0 (untrained) | step 25 | step 100 |
|---|---:|---:|---:|
| LitBench | 99.6% | 99.0% (Phase 2) | 93.5% (Phase 2) |
| HelpSteer3 | 98.2% | 99.0% | 89.2% |
| Arena | 98.5% | 96.2% | 97.1% |

Training tends to *reduce* late-stage validity (HS3: 99→89; LitBench Phase 2: 99→93). Phase 5 (criteria_margin) bucks the trend (99.5%).

### Criterion count and weight entropy

| Run group | Criteria/rubric (mean) | Weight entropy (bits) |
|---|---:|---:|
| Untrained (all 3 datasets) | 4.7-4.9 | 2.17-2.26 |
| Trained step 25 | 4.9-5.0 | 2.20-2.28 |
| Trained step 100 | 3.9-4.1 | 1.83-1.90 |

**Training concentrates weight on fewer criteria.** This is the most consistent training effect. Untrained policies write ~5 criteria with relatively spread weights; trained-step-100 policies converge on 3-4 criteria with concentrated weight.

### Vocabulary collapse — the most striking finding

Top criterion-name bigrams change dramatically with training:

**LitBench Phase 2:**
- Untrained: "prompt fidelity" (683), "narrative coherence" (555), "character development" (332), "originality creativity" (169)
- Step 100: **"emotional resonance" (197), "emotional engagement" (189), "emotional authenticity" (171), "emotional impact" (156)** — `emotional X` dominates; "prompt fidelity" disappears.

**HelpSteer3:**
- Untrained: "prompt fidelity" (154), "relevance prompt" (69), "clarity structure" (54)
- Step 100: "relevance focus" (35), "clarity conciseness" (25), "clarity organization" (24) — moves from "prompt fidelity" to "relevance/clarity" framings.

**Arena:**
- Untrained: "prompt fidelity" (223), "clarity structure" (206), "relevance prompt" (165)
- Step 100: "clarity structure" (128), "relevance focus" (125), "clarity organization" (66) — drops "prompt fidelity," concentrates on "clarity X."

**Pattern across all three datasets:** training de-emphasizes "prompt fidelity" (a domain-agnostic baseline criterion) and converges on dataset-specific criterion templates. Whether this is reward hacking on the in-house judge or a genuine refinement is what the cross-judge gap analysis is telling us.

## 4. What this means for the writeup

**Don't claim:** "RuPO improves preference accuracy over base models." The data doesn't support it. Base Qwen3-8B already writes effective rubrics (instruction-tuning + the structured prompt template do most of the work).

**Do claim, in order of evidence strength:**

1. **(Strong)** "Training under one judge can produce rubrics that a held-out judge rates more highly than the training judge does." Phase 5's −0.085 gap with significance under both GPT-5 and Sonnet supports this. The interpretation is that `criteria_margin`-trained rubrics emphasize features the in-house Qwen judge under-weights.

2. **(Medium)** "Training induces a measurable shift in rubric content — fewer criteria, more concentrated weights, dataset-specific vocabulary preferences (away from 'prompt fidelity' toward dataset-relevant criteria)." The rubric content analysis shows this consistently across all 3 datasets.

3. **(Medium)** "The method generalizes structurally — training proceeds without instability across creative writing, helpfulness, and general chat domains, with each dataset finding a different optimal stopping step (LitBench peaks late, HS3 and Arena peak early)."

4. **(Weak/honest caveat)** "Training does not improve preference accuracy on the in-house judge across any of three preference datasets at the 100-step horizon. The training signal seems to optimize for properties orthogonal to the metric being measured."

The honest framing for the paper: **RuPO is a method for *steering* rubric content, not for improving raw preference accuracy.** The cross-judge gap is evidence of judge-specific shift, not improvement.

## 5. Open questions (for revisions)

- **Does early-stopping help?** The trained step-25 numbers (HS3: 0.693, Arena: 0.575) are at parity with untrained — but the trained model has different rubric content. With the right stopping point, you might get equal accuracy with the desired content shift. This is the most promising follow-up.
- **Does this hold on n>1 seeds?** seed=1 (5uwbvhqf) was the lucky seed; seed=2 (Phase 2) was bad; seed=3 was degenerate. The story is fragile across seeds.
- **What happens with `criteria_margin` on HS3/Arena?** Phase 5's `criteria_margin` showed the strongest cross-judge gap. If this generalizes to HS3/Arena, it'd cement the "shape matters for judge-portability" claim.

## 6. Cost summary

- API spend: ~$200 (Phase 5 cross-judges + matched + HS3 + Arena cross-judges)
- GPU time: ~9.5h of the 16h budget
- Two retries for crashes (HS3 v1 stuck on big val, Arena v1+v2 OOM on long prompts) cost ~80 min combined

## File pointers

- All logs: `/root/rubric-policy-optimization/SDPO/logs/`
- Cross-judge JSONs: `logs/seed2_constantlr_total_absolute_20260504-071351/crossjudge_*.json`, `logs/helpsteer3_*/crossjudge_*.json`, `logs/arena_*/crossjudge_*.json`
- Untrained baseline val_generations: `logs/untrained_baseline_*/val_generations/0.jsonl`
- Rubric content analysis: `/tmp/rubric_analysis.txt` and `/tmp/rubric_analysis.json`
- Bootstrap CIs script: `scripts/bootstrap_crossjudge_cis.py`
- Rubric analyzer: `scripts/analyze_rubric_quality.py`
- HelpSteer3 conversion: `scripts/convert_helpsteer3_to_dpo.py`
