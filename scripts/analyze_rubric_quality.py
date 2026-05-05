"""Analyze rubric content across runs.

Reads policy outputs (rubrics) from any of:
  - val_generations JSONL files (verl trainer.validation_data_dir output)
  - cross-judge result JSONs (rows[].policy_output)

Computes, per source:
  - Criterion count distribution (mean, median, stdev, range)
  - Criterion weight distribution and entropy (does the policy spread weight or concentrate it?)
  - Rubric length (chars, criterion count)
  - Top criterion-name unigrams + bigrams (what concepts does the policy emphasize?)
  - Validity rate (rubrics with criterion_count>=2 and total_weight==100)

Output is a markdown table to stdout. Pass --json to also dump structured stats.

Usage:
  python scripts/analyze_rubric_quality.py \
      --source val_generations:logs/.../val_generations/100.jsonl:label1 \
      --source crossjudge:logs/.../crossjudge_*.json:label2 \
      [--source ...] \
      [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# Bootstrap path so we can import the env's rubric parser.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "environments" / "dpo_to_rupo_verl"))
sys.path.insert(0, str(_HERE.parent.parent / "environments" / "dpo_to_rupo"))

from dpo_to_rupo.structured_rubric import inspect_structured_rubric
from rubric_parser import extract_rubric_text


_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "by",
    "as", "at", "is", "are", "be", "this", "that", "it", "its", "from", "into",
    "any", "all", "no", "not", "but", "if", "than", "vs", "vs.", "etc",
}


def tokenize(name: str) -> list[str]:
    """Lowercased word tokens, stripped of stopwords."""
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]+", name.lower()) if t not in _STOPWORDS]


def shannon_entropy(weights: list[float]) -> float:
    """Entropy of normalized weights, in bits. 0 = single criterion gets all weight."""
    s = sum(weights)
    if s <= 0: return 0.0
    p = [w / s for w in weights if w > 0]
    return -sum(pi * math.log2(pi) for pi in p)


def load_outputs(source_spec: str) -> tuple[str, str, list[str]]:
    """Parse `kind:path:label` and return (kind, label, [rubric_text, ...])."""
    parts = source_spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"--source must be `kind:path:label`, got {source_spec!r}")
    kind, path, label = parts
    rubrics: list[str] = []
    if kind == "val_generations":
        n_bad = 0
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            out = d.get("output", "")
            r = extract_rubric_text(out)
            if r is not None:
                rubrics.append(r)
            else:
                rubrics.append("")  # keep failed parses for validity-rate calc
        if n_bad:
            print(f"  ({path}: skipped {n_bad} malformed JSONL lines)")
    elif kind == "crossjudge":
        d = json.load(open(path))
        for r in d["rows"]:
            out = r.get("policy_output", "")
            rt = extract_rubric_text(out)
            rubrics.append(rt or "")
    else:
        raise SystemExit(f"Unknown source kind: {kind!r}")
    return kind, label, rubrics


def analyze_one(label: str, rubrics: list[str]) -> dict:
    """Compute summary stats for a list of rubric texts."""
    n_total = len(rubrics)
    n_parse = sum(1 for r in rubrics if r)
    valid_rubrics = []
    inspections = []
    for r in rubrics:
        if not r:
            continue
        ins = inspect_structured_rubric(r)
        inspections.append(ins)
        if ins.valid:
            valid_rubrics.append(ins)
    n_valid = len(valid_rubrics)

    if not valid_rubrics:
        return {
            "label": label,
            "n_total": n_total, "n_parse": n_parse, "n_valid": n_valid,
            "criterion_counts": [], "weights_per_rubric": [], "entropies": [],
            "rubric_chars": [], "criterion_name_tokens": [],
        }

    criterion_counts = [len(ins.criteria) for ins in valid_rubrics]
    weights_per_rubric = [[c.weight for c in ins.criteria] for ins in valid_rubrics]
    entropies = [shannon_entropy(w) for w in weights_per_rubric]
    rubric_chars = [sum(len(c.name) + len(c.description) for c in ins.criteria) for ins in valid_rubrics]
    name_tokens = []
    name_bigrams = []
    for ins in valid_rubrics:
        for c in ins.criteria:
            toks = tokenize(c.name)
            name_tokens.extend(toks)
            name_bigrams.extend(f"{a} {b}" for a, b in zip(toks, toks[1:]))

    return {
        "label": label,
        "n_total": n_total,
        "n_parse": n_parse,
        "n_valid": n_valid,
        "validity_rate": n_valid / n_total if n_total else float("nan"),
        "criterion_counts": criterion_counts,
        "weights_per_rubric": weights_per_rubric,
        "entropies": entropies,
        "rubric_chars": rubric_chars,
        "criterion_name_tokens": name_tokens,
        "criterion_name_bigrams": name_bigrams,
    }


def fmt_dist(values: list[float], digits: int = 2) -> str:
    if not values:
        return "n/a"
    fmt = f"{{:.{digits}f}}"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.fmean(values))} ± {fmt.format(statistics.pstdev(values))} (n={len(values)})"


def report(stats_list: list[dict], top_k: int = 12) -> None:
    print(f"\n## Rubric content analysis ({len(stats_list)} runs)\n")

    print("### Validity")
    print(f"{'label':40s} {'n':>5s} {'parse':>10s} {'valid':>10s} {'valid%':>8s}")
    print("-" * 80)
    for s in stats_list:
        n = s["n_total"]
        rate = s["n_valid"] / n if n else 0
        print(f"{s['label'][:40]:40s} {n:>5d} {s['n_parse']:>10d} {s['n_valid']:>10d} {rate*100:>7.1f}%")

    print("\n### Criterion counts (per valid rubric)")
    print(f"{'label':40s} {'count mean ± std':>30s} {'min':>5s} {'max':>5s}")
    print("-" * 80)
    for s in stats_list:
        c = s["criterion_counts"]
        if c:
            print(f"{s['label'][:40]:40s} {fmt_dist(c, 1):>30s} {min(c):>5d} {max(c):>5d}")
        else:
            print(f"{s['label'][:40]:40s} {'n/a (no valid rubrics)':>30s}")

    print("\n### Weight distribution (entropy in bits, 0 = all weight on 1 criterion)")
    print(f"{'label':40s} {'entropy mean ± std':>32s}")
    print("-" * 80)
    for s in stats_list:
        e = s["entropies"]
        print(f"{s['label'][:40]:40s} {fmt_dist(e, 3):>32s}")

    print("\n### Rubric length (sum of name+desc chars per rubric)")
    print(f"{'label':40s} {'chars mean ± std':>30s}")
    print("-" * 80)
    for s in stats_list:
        c = s["rubric_chars"]
        print(f"{s['label'][:40]:40s} {fmt_dist(c, 0):>30s}")

    print("\n### Top criterion-name unigrams (top-{} per run)".format(top_k))
    for s in stats_list:
        ctr = Counter(s["criterion_name_tokens"]).most_common(top_k)
        print(f"  {s['label']}: {', '.join(f'{w}×{c}' for w, c in ctr)}")

    print("\n### Top criterion-name bigrams (top-{} per run)".format(top_k))
    for s in stats_list:
        ctr = Counter(s["criterion_name_bigrams"]).most_common(top_k)
        if ctr:
            print(f"  {s['label']}: {', '.join(f'\"{w}\"×{c}' for w, c in ctr)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="`kind:path:label`. kind=val_generations|crossjudge. Repeatable.")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--json", help="Also dump full per-run stats as JSON to this path.")
    args = ap.parse_args()

    stats_list: list[dict] = []
    for spec in args.source:
        kind, label, rubrics = load_outputs(spec)
        s = analyze_one(label, rubrics)
        stats_list.append(s)

    report(stats_list, args.top_k)

    if args.json:
        # Strip large lists from the JSON dump (keep distributions but as summary stats only).
        compact = []
        for s in stats_list:
            c = dict(s)
            for k in ("criterion_counts", "weights_per_rubric", "entropies", "rubric_chars",
                      "criterion_name_tokens", "criterion_name_bigrams"):
                v = s.get(k, [])
                if v and isinstance(v[0], (int, float)):
                    c[k] = {"n": len(v), "mean": statistics.fmean(v), "stdev": statistics.pstdev(v) if len(v) > 1 else 0.0,
                            "min": min(v), "max": max(v)}
                elif k.endswith("tokens") or k.endswith("bigrams"):
                    c[k] = Counter(v).most_common(50)
                else:
                    c[k] = "<lists omitted>"
            compact.append(c)
        Path(args.json).write_text(json.dumps(compact, indent=2))
        print(f"\nWrote stats to {args.json}")


if __name__ == "__main__":
    main()
