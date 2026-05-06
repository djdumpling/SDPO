"""Phase 7 — minimal-prompt baseline (H11 falsifier).

Asks Qwen3-8B "which response is better, A or B?" with NO rubric, NO scaffolding.
Computes the resulting preference accuracy on a val_generations JSONL (we use
the gts field for chosen/rejected ground truth; we don't need the policy output).

Headline: if minimal-prompt accuracy ≥ 0.65, the rubric template wasn't carrying
much weight (training has very limited headroom). If it drops to ≤0.55, the
rubric template is doing most of the lift and accuracy ceiling is near-binding.
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

from openai import AsyncOpenAI


SYS = (
    "You are a helpful evaluator. You will be shown a prompt and two candidate "
    "responses, A and B. Pick the better response. Reply with exactly one letter: "
    "A or B. No explanation."
)


def build_user(prompt, a, b):
    return f"""Prompt:
{prompt}

Response A:
{a}

Response B:
{b}

Which response is better? Answer with just one letter, A or B."""


_LETTER_RE = re.compile(r"\b([AB])\b", re.I)
_THINK_RE = re.compile(r"</think>", re.I)


def parse_letter(text):
    if not text: return None
    s = text.strip()
    # If thinking was emitted, only look at content after </think>.
    m_think = _THINK_RE.search(s)
    if m_think:
        s = s[m_think.end():].strip()
    m = _LETTER_RE.search(s)
    return m.group(1).upper() if m else None


async def call(client, sema, model, max_tokens, user, temperature):
    async with sema:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return parse_letter(resp.choices[0].message.content)
        except Exception:
            return None


async def run(args):
    rng = random.Random(args.seed)
    rows = []
    for ln in Path(args.jsonl).read_text().splitlines():
        if not ln.strip(): continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError: continue
        gts_raw = d.get("gts")
        if not isinstance(gts_raw, str): continue
        try: g = json.loads(gts_raw)
        except json.JSONDecodeError: continue
        prompt = (g.get("prompt") or "").strip()
        chosen = (g.get("chosen") or "").strip()
        rejected = (g.get("rejected") or "").strip()
        if not (prompt and chosen and rejected): continue
        # Randomize position to avoid position bias confound
        if rng.random() < 0.5:
            a_text, b_text, correct = chosen, rejected, "A"
        else:
            a_text, b_text, correct = rejected, chosen, "B"
        rows.append({"prompt": prompt, "a": a_text, "b": b_text, "correct": correct})

    if args.limit and len(rows) > args.limit:
        rng.shuffle(rows); rows = rows[: args.limit]
    print(f"  {len(rows)} prompts to evaluate")

    client = AsyncOpenAI(api_key=os.environ.get(args.api_key_env, "x"),
                         base_url=args.base_url, timeout=120.0)
    sema = asyncio.Semaphore(args.max_concurrent)
    tasks = [call(client, sema, args.model, args.max_tokens,
                  build_user(r["prompt"], r["a"], r["b"]), args.temperature) for r in rows]
    answers = await asyncio.gather(*tasks)

    correct = 0
    n_parsed = 0
    out_rows = []
    for r, a in zip(rows, answers):
        if a in ("A", "B"):
            n_parsed += 1
            ok = (a == r["correct"])
            correct += int(ok)
            out_rows.append({"correct_letter": r["correct"], "answered": a, "ok": ok})
        else:
            out_rows.append({"correct_letter": r["correct"], "answered": None, "ok": None})

    acc = correct / n_parsed if n_parsed else None
    headline = {
        "n_total": len(rows),
        "n_parsed": n_parsed,
        "minimal_prompt_accuracy": acc,
        "verdict": None,
    }
    if acc is None:
        headline["verdict"] = "Could not parse responses; check model/server."
    elif acc >= 0.65:
        headline["verdict"] = (
            f"H11 SUPPORTED — Qwen3-8B answers minimal-prompt 'which is better' at "
            f"acc={acc:.3f}, ≥0.65 (similar to ~0.69 with full rubric). Rubric template "
            f"is contributing little headroom; RuPO has very little room to improve.")
    elif acc <= 0.55:
        headline["verdict"] = (
            f"H11 RULED OUT — minimal-prompt acc={acc:.3f} is much lower than 0.69 with "
            f"rubric. The rubric template carries most of the signal; trained policies "
            f"plausibly should be able to add value on top.")
    else:
        headline["verdict"] = (
            f"H11 partial — minimal-prompt acc={acc:.3f}. Rubric template contributes "
            f"some lift but is not the only source.")

    out = {"headline": headline, "rows": out_rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(headline, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="LOCAL_DUMMY_KEY")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
