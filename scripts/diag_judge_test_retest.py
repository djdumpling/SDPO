"""Phase 3 — judge test-retest (H1 falsifier).

Re-judges a fixed set of (rubric, response) pairs k times each and reports per-pair
test-retest variance vs between-pair variance. If test-retest σ ≥ between-pair σ,
the judge SNR is ≤ 1 and the optimization landscape is dominated by judge noise.

Subsamples N triples from a val_generations JSONL, scores chosen+rejected on each
criterion k times via the local Qwen3-8B vLLM, computes per-triple variance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

# Reuse SDPO env helpers
_HERE = Path(__file__).resolve().parent
_SDPO_ROOT = _HERE.parent
_REPO_ROOT = _SDPO_ROOT.parent
sys.path.insert(0, str(_SDPO_ROOT / "environments" / "dpo_to_rupo_verl"))
sys.path.insert(0, str(_REPO_ROOT / "environments" / "dpo_to_rupo"))

from openai import AsyncOpenAI
from dpo_to_rupo.structured_rubric import inspect_structured_rubric
from rubric_parser import extract_rubric_text, extract_score


JUDGE_SYS = """You are an expert grader of creative-writing stories against scoring rubrics.

You will be given:
- a writing prompt,
- one candidate story written for that prompt,
- and a scoring rubric for evaluating that story on a 0-100 scale.

Your job is to grade the story against the rubric and return a single integer from 0 to 100, where 100 means the story satisfies the rubric extremely well."""


def build_criterion_prompt(prompt_text, story_text, idx, name, desc):
    return f"""You are grading one response on one rubric criterion.

Here is the prompt:
<prompt>
{prompt_text}
</prompt>

Here is the response:
<response>
{story_text}
</response>

Here is the criterion:
<criterion>
<index>{idx}</index>
<name>{name}</name>
<description>{desc}</description>
</criterion>

Assign one integer score from 0 to 100 for this criterion only.
- Judge only this criterion.
- Use the full range when appropriate.
- Do not infer or mention any total score.
- Ignore any rubric weighting. Criterion weights are applied later outside this grading call.

Then return:
<analysis>
Short justification grounded in the criterion and the response.
</analysis>
<score>
An integer from 0 to 100.
</score>

Return only those XML tags.
"""


async def call_judge(client, sema, model, max_tokens, user, temperature):
    async with sema:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content
            return extract_score(text)
        except Exception as e:
            return None


async def run(args):
    rng = random.Random(args.seed)
    rows = [json.loads(ln) for ln in Path(args.jsonl).read_text().splitlines() if ln.strip()]
    print(f"  loaded {len(rows)} val rows")
    rng.shuffle(rows)

    client = AsyncOpenAI(api_key=os.environ.get(args.judge_api_key_env, "x"),
                         base_url=args.judge_base_url, timeout=300.0)
    sema = asyncio.Semaphore(args.max_concurrent)

    triples_data = []  # one per (rubric, criterion_idx, response_kind)
    n_kept = 0
    for row in rows:
        if n_kept >= args.n_triples:
            break
        rubric = extract_rubric_text(row.get("output", ""))
        if not rubric:
            continue
        ins = inspect_structured_rubric(rubric)
        if not ins.valid or not ins.criteria:
            continue
        gts = row.get("gts") or {}
        if isinstance(gts, str):
            try: gts = json.loads(gts)
            except json.JSONDecodeError: continue
        prompt = gts.get("prompt", "")
        chosen = gts.get("chosen", "")
        rejected = gts.get("rejected", "")
        if not (prompt and chosen and rejected):
            continue
        # First criterion only (one judge call per response saves runtime; H1 is about per-call σ regardless)
        c = ins.criteria[0]
        triples_data.append({
            "row_idx": rows.index(row),
            "prompt": prompt, "rubric": rubric, "chosen": chosen, "rejected": rejected,
            "criterion_idx": 1, "criterion_name": c.name, "criterion_desc": c.description,
        })
        n_kept += 1

    print(f"  prepared {n_kept} triples; calling judge k={args.k_resamples} times each")
    # K resamples × 2 responses × N triples calls
    tasks = []
    task_meta = []
    for ti, t in enumerate(triples_data):
        for kk in range(args.k_resamples):
            for kind, resp in [("chosen", t["chosen"]), ("rejected", t["rejected"])]:
                user = build_criterion_prompt(
                    t["prompt"], resp, t["criterion_idx"], t["criterion_name"], t["criterion_desc"])
                tasks.append(call_judge(
                    client, sema, args.judge_model, args.judge_max_tokens, user, args.temperature))
                task_meta.append((ti, kk, kind))

    print(f"  total judge calls: {len(tasks)}")
    results = await asyncio.gather(*tasks)
    # Index back
    per_triple = {}
    for (ti, kk, kind), score in zip(task_meta, results):
        d = per_triple.setdefault(ti, {"chosen": [], "rejected": []})
        d[kind].append(score)

    rows_out = []
    chosen_stds = []
    rejected_stds = []
    margins_mean = []
    margins_std = []
    for ti, t in enumerate(triples_data):
        ch = [s for s in per_triple.get(ti, {}).get("chosen", []) if s is not None]
        rj = [s for s in per_triple.get(ti, {}).get("rejected", []) if s is not None]
        margin_samples = [(c - r) for c, r in zip(ch, rj)] if (len(ch) == len(rj) and ch) else []
        ch_mu = statistics.fmean(ch) if ch else None
        rj_mu = statistics.fmean(rj) if rj else None
        ch_sd = statistics.pstdev(ch) if len(ch) > 1 else None
        rj_sd = statistics.pstdev(rj) if len(rj) > 1 else None
        m_mu = statistics.fmean(margin_samples) if margin_samples else None
        m_sd = statistics.pstdev(margin_samples) if len(margin_samples) > 1 else None

        rows_out.append({
            "ti": ti,
            "criterion": t["criterion_name"],
            "chosen_scores": ch, "rejected_scores": rj,
            "chosen_mean": ch_mu, "chosen_std": ch_sd,
            "rejected_mean": rj_mu, "rejected_std": rj_sd,
            "margin_mean": m_mu, "margin_std": m_sd,
        })
        if ch_sd is not None: chosen_stds.append(ch_sd)
        if rj_sd is not None: rejected_stds.append(rj_sd)
        if m_mu is not None:  margins_mean.append(m_mu)
        if m_sd is not None:  margins_std.append(m_sd)

    headline = {
        "n_triples": len(rows_out),
        "k_resamples": args.k_resamples,
        "judge_temperature": args.temperature,
        # Test-retest (within-triple) — judge noise σ
        "test_retest_chosen_sigma_mean": statistics.fmean(chosen_stds) if chosen_stds else None,
        "test_retest_rejected_sigma_mean": statistics.fmean(rejected_stds) if rejected_stds else None,
        "test_retest_margin_sigma_mean": statistics.fmean(margins_std) if margins_std else None,
        # Between-triple — what the optimizer is trying to detect
        "between_triple_chosen_sigma": statistics.pstdev(
            [r["chosen_mean"] for r in rows_out if r["chosen_mean"] is not None]) if rows_out else None,
        "between_triple_margin_sigma": statistics.pstdev(margins_mean) if len(margins_mean) > 1 else None,
        "between_triple_margin_mean": statistics.fmean(margins_mean) if margins_mean else None,
        "verdict": None,
    }

    tr_sigma = headline["test_retest_margin_sigma_mean"] or 0.0
    bw_sigma = headline["between_triple_margin_sigma"] or 0.0
    bw_mean = abs(headline["between_triple_margin_mean"] or 0.0)
    if tr_sigma > 0:
        snr = bw_mean / tr_sigma
        headline["snr_margin_mean_over_test_retest_sigma"] = snr
        if snr < 1:
            headline["verdict"] = (
                f"H1 CONFIRMED — judge per-call σ on margin = {tr_sigma:.1f}pt, "
                f"between-triple mean margin = {bw_mean:.1f}pt → SNR = {snr:.2f}. "
                f"Judge noise dominates the rubric signal. k=4 ensembling should "
                f"reduce σ to ~{tr_sigma/2:.1f}pt and restore signal.")
        elif snr < 2:
            headline["verdict"] = (
                f"H1 PARTIALLY CONFIRMED — judge σ = {tr_sigma:.1f}pt, signal = {bw_mean:.1f}pt, "
                f"SNR = {snr:.2f}. Ensembling will help materially.")
        else:
            headline["verdict"] = (
                f"H1 RULED OUT — judge σ = {tr_sigma:.1f}pt, signal = {bw_mean:.1f}pt, "
                f"SNR = {snr:.2f}. Judge noise is not the dominant pathology.")

    out = {"headline": headline, "rows": rows_out}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\nHEADLINE:")
    print(json.dumps(headline, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--judge-base-url", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-api-key-env", default="LOCAL_DUMMY_KEY")
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument("--n-triples", type=int, default=100)
    ap.add_argument("--k-resamples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
