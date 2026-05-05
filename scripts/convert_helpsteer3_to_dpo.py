"""Convert nvidia/HelpSteer3 → DPO-style {prompt, chosen, rejected} HF dataset.

Saved locally as an HF DatasetDict so SDPO/environments/dpo_to_rupo_verl/preprocess_dataset.py
can consume it via load_dataset(local_path).

HelpSteer3 schema:
  context              : list[{role, content}] — chat history (1+ turns)
  response1, response2 : alternative final-assistant responses
  overall_preference   : int in {-3,-2,-1,1,2,3}; <0 → response1 preferred, >0 → response2 preferred

We:
  - Serialize `context` as a single multi-turn prompt string
  - Set chosen = preferred response, rejected = the other
  - Drop rows with overall_preference == 0 (ties), if any
  - Keep all 4 domains (general, code, multilingual, stem) so the headline result
    spans diverse content, not just one type

Output: an HF dataset directory with `train` and `test` splits, columns
(prompt, chosen, rejected). The `test` split is HelpSteer3's `validation` split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def serialize_chat(context: list[dict]) -> str:
    """Render a chat list as `<role>:\\n<content>\\n\\n` blocks. Mirrors how a model
    would receive it via a chat template, but in plain text so any prompt mode
    can splice it in."""
    parts = []
    for msg in context:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts).strip()


def convert_split(ds, split_label: str):
    """Map HelpSteer3 rows to {prompt, chosen, rejected}."""
    def _row(ex):
        pref = int(ex["overall_preference"])
        if pref == 0:
            # No preference; will be filtered downstream.
            return {"prompt": "", "chosen": "", "rejected": ""}
        prompt = serialize_chat(ex["context"])
        if pref < 0:
            chosen, rejected = ex["response1"], ex["response2"]
        else:
            chosen, rejected = ex["response2"], ex["response1"]
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    out = ds.map(_row, remove_columns=ds.column_names)
    out = out.filter(lambda r: r["prompt"] and r["chosen"] and r["rejected"])
    print(f"  {split_label}: {len(ds)} → {len(out)} after filtering ties/empties")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="nvidia/HelpSteer3")
    ap.add_argument("--out-dir", required=True, help="Directory to save the converted DatasetDict.")
    args = ap.parse_args()

    print(f"Loading {args.source}...")
    train = load_dataset(args.source, split="train")
    val = load_dataset(args.source, split="validation")
    print(f"  train: {len(train)}, validation: {len(val)}")

    train_out = convert_split(train, "train")
    test_out = convert_split(val, "test")

    # Save parquet files with the standard HF auto-detect layout so
    # `datasets.load_dataset(out_dir, split="train")` works directly. The
    # filename pattern `<split>.parquet` is auto-discovered by load_dataset.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train_out.to_parquet(str(train_path))
    test_out.to_parquet(str(test_path))

    print(f"\nSaved DPO-style dataset to {out_dir}")
    print(f"  train.parquet: {len(train_out)} rows")
    print(f"  test.parquet:  {len(test_out)} rows")
    print(f"\nLoad later with:")
    print(f"  datasets.load_dataset(\"{out_dir}\", split=\"train\")")


if __name__ == "__main__":
    main()
