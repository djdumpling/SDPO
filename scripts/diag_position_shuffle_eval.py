"""Phase 6 — position-shuffle re-eval (H7 falsifier).

Swaps `chosen` <-> `rejected` in a val_generations JSONL and re-runs the cross-
judge over the flipped pairs using the local Qwen3-8B endpoint. If the model
has no position bias, flipped accuracy should be (1 - original_accuracy). If it
has position bias toward the "chosen" slot, flipped accuracy will be ≈
original. Reports both numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--judge-base-url", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-api-key-env", default="LOCAL_DUMMY_KEY")
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    # Flip chosen <-> rejected in gts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        flipped_path = Path(f.name)
        n_in = 0
        n_out = 0
        for line in src.read_text().splitlines():
            if not line.strip(): continue
            n_in += 1
            d = json.loads(line)
            gts_raw = d.get("gts")
            if isinstance(gts_raw, str):
                try:
                    g = json.loads(gts_raw)
                except json.JSONDecodeError:
                    continue
                g["chosen"], g["rejected"] = g.get("rejected", ""), g.get("chosen", "")
                d["gts"] = json.dumps(g)
                f.write(json.dumps(d) + "\n")
                n_out += 1

    print(f"  flipped {n_out}/{n_in} rows  → {flipped_path}")
    flipped_out = Path(args.out)
    flipped_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(Path(__file__).parent / "cross_judge_rubric.py"),
           "--source", "jsonl", "--jsonl", str(flipped_path),
           "--judge-base-url", args.judge_base_url,
           "--judge-model", args.judge_model,
           "--judge-api-key-env", args.judge_api_key_env,
           "--temperature", "0.7",
           "--max-concurrent", str(args.max_concurrent),
           "--output", str(flipped_out)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    print("  running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"  cross_judge returned rc={rc}")
        return

    # Annotate with verdict
    d = json.loads(flipped_out.read_text())
    rows = d.get("rows", [])
    rewards = [r.get("reward") for r in rows if r.get("reward") is not None]
    if rewards:
        flipped_acc = sum(rewards) / len(rewards)
        verdict = None
        if flipped_acc > 0.6:
            verdict = (
                f"H7 CONFIRMED — flipped accuracy = {flipped_acc:.3f}. The judge picks "
                f"whatever is in the 'chosen' slot regardless of content. STRONG position bias.")
        elif flipped_acc < 0.4:
            verdict = (
                f"H7 RULED OUT — flipped accuracy = {flipped_acc:.3f} ≈ 1 - original_accuracy. "
                f"Judge correctly identifies quality regardless of position; no position bias.")
        else:
            verdict = (
                f"H7 partial — flipped accuracy = {flipped_acc:.3f}. Some position-bias "
                f"contribution but not the dominant pathology.")
        d["headline"] = {"flipped_accuracy": flipped_acc, "verdict": verdict, "n_rows": len(rewards)}
        flipped_out.write_text(json.dumps(d, indent=2))
        print("\nHEADLINE:", verdict)


if __name__ == "__main__":
    main()
