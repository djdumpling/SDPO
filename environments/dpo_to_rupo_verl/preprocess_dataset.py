"""Convert a HuggingFace DPO preference dataset to verl's parquet format.

Usage:
    python -m environments.dpo_to_rupo_verl.preprocess_dataset \
        --dataset_name sumuks/litbench-ha-with-baseline \
        --policy_prompt_mode pair \
        --judge_reward_type margin \
        --output_dir datasets/dpo_to_rupo

Produces train.parquet and test.parquet with verl's expected schema:
    data_source, prompt, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import json
import os
import random

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

# Import prompt content and builders from the installed dpo_to_rupo package.
# Falls back to path-based import if the package is not installed.
try:
    from dpo_to_rupo.prompts import (
        CRITERIA_RUBRIC_FORMAT_APPENDIX,
        DEFAULT_PAIR_SYSTEM_PROMPT,
        DEFAULT_PROMPT_ONLY_SYSTEM_PROMPT,
        build_policy_prompt,
    )
    from dpo_to_rupo.config import CRITERIA_REWARD_TYPES
except ImportError:
    import sys
    _env_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")), "environments", "dpo_to_rupo")
    if _env_path not in sys.path:
        sys.path.insert(0, _env_path)
    from dpo_to_rupo.prompts import (
        CRITERIA_RUBRIC_FORMAT_APPENDIX,
        DEFAULT_PAIR_SYSTEM_PROMPT,
        DEFAULT_PROMPT_ONLY_SYSTEM_PROMPT,
        build_policy_prompt,
    )
    from dpo_to_rupo.config import CRITERIA_REWARD_TYPES


# ---------------------------------------------------------------------------
# Parquet helpers copied from SDPO/data/preprocess.py to handle LargeString.
# ---------------------------------------------------------------------------

def _to_large(field: pa.Field) -> pa.Field:
    t = field.type
    if pa.types.is_string(t):
        return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(t):
        return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(t):
        return pa.field(
            field.name,
            pa.large_list(_to_large(pa.field("item", t.value_type)).type),
            field.nullable,
            field.metadata,
        )
    if pa.types.is_struct(t):
        return pa.field(
            field.name,
            pa.struct([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in t]),
            field.nullable,
            field.metadata,
        )
    return field


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema])


def write_rowgrouped_large(ds: datasets.Dataset, path: str, rows_per_group: int = 32) -> None:
    """Cast to LargeString/LargeList and write many small row groups."""
    tbl: pa.Table = ds.data.table
    tbl = tbl.cast(_large_schema(tbl.schema))
    n = len(tbl)
    writer = None
    try:
        for start in range(0, n, rows_per_group):
            chunk = tbl.slice(start, min(rows_per_group, n - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------

THINKING_SYSTEM_PROMPT_APPENDIX = """

Thinking-mode output contract
- Use the model's thinking channel for private scratch work when it is available.
- Close any `<think>` block before emitting the final XML.
- The final answer must contain exactly one closed `<analysis>...</analysis>` block followed by one closed `<rubric>...</rubric>` block.
- Keep the final `<analysis>` concise. Put detailed private reasoning in the thinking channel, not in the final XML.
- Always finish the closing `</rubric>` tag before the response ends.
"""


def _parse_bool(value: str | bool) -> bool:
    """Parse shell-friendly boolean values from CLI arguments."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _make_thinking_aware_system_prompt(system_prompt: str) -> str:
    """Adjust the policy prompt so hidden thinking and final XML do not compete."""
    private_analysis_instruction = "1. First reason privately inside `<analysis>...</analysis>`."
    final_analysis_instruction = (
        "1. Use the model's thinking channel for private scratch work if it is available. "
        "After thinking, write a concise public `<analysis>...</analysis>` summary."
    )
    system_prompt = system_prompt.replace(private_analysis_instruction, final_analysis_instruction)
    return system_prompt.rstrip() + THINKING_SYSTEM_PROMPT_APPENDIX


def _get_system_prompt(
    policy_prompt_mode: str,
    judge_reward_type: str,
    custom_system_prompt: str | None,
    enable_thinking_prompt: bool,
) -> str:
    """Return the system prompt for the policy model."""
    if custom_system_prompt:
        base = custom_system_prompt
    else:
        base = DEFAULT_PAIR_SYSTEM_PROMPT if policy_prompt_mode == "pair" else DEFAULT_PROMPT_ONLY_SYSTEM_PROMPT

    if judge_reward_type in CRITERIA_REWARD_TYPES:
        base = base + CRITERIA_RUBRIC_FORMAT_APPENDIX
    return _make_thinking_aware_system_prompt(base) if enable_thinking_prompt else base


def make_map_fn(
    split: str,
    policy_prompt_mode: str,
    judge_reward_type: str,
    randomize_order: bool,
    system_prompt: str,
):
    """Return a row mapping function for datasets.map()."""

    def process_fn(example: dict, idx: int) -> dict:
        prompt_text = str(example["prompt"])
        chosen_text = str(example["chosen"])
        rejected_text = str(example["rejected"])

        # Randomize candidate A/B order in pair mode to prevent position bias.
        swap = policy_prompt_mode == "pair" and randomize_order and random.random() < 0.5
        candidate_a = rejected_text if swap else chosen_text
        candidate_b = chosen_text if swap else rejected_text

        user_content = build_policy_prompt(prompt_text, candidate_a, candidate_b, policy_prompt_mode)

        # Build ground truth payload. The reward function uses this to recover
        # the preference pair for judge scoring.
        ground_truth = json.dumps(
            {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text},
            separators=(",", ":"),
        )

        return {
            "data_source": "dpo_to_rupo",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "ability": "rubric",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": str(idx),
                "policy_prompt_mode": policy_prompt_mode,
                "judge_reward_type": judge_reward_type,
                "candidate_a_is_chosen": not swap,
            },
        }

    return process_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess(
    dataset_name: str,
    policy_prompt_mode: str = "pair",
    judge_reward_type: str = "margin",
    randomize_order: bool = True,
    custom_system_prompt: str | None = None,
    enable_thinking_prompt: bool = False,
    output_dir: str = "datasets/dpo_to_rupo",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    system_prompt = _get_system_prompt(policy_prompt_mode, judge_reward_type, custom_system_prompt, enable_thinking_prompt)

    for split in ("train", "test"):
        print(f"Loading {dataset_name} split={split}...")
        ds = datasets.load_dataset(dataset_name, split=split)
        print(f"  {len(ds)} rows")

        map_fn = make_map_fn(split, policy_prompt_mode, judge_reward_type, randomize_order, system_prompt)
        ds = ds.map(map_fn, with_indices=True, remove_columns=ds.column_names)

        out_path = os.path.join(output_dir, f"{split}.parquet")
        write_rowgrouped_large(ds, out_path)
        print(f"  Wrote {out_path} ({len(ds)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DPO datasets to verl parquet for SDPO rubric training.")
    parser.add_argument("--dataset_name", type=str, default="sumuks/litbench-ha-with-baseline", help="HuggingFace dataset name.")
    parser.add_argument("--policy_prompt_mode", type=str, default="pair", choices=["pair", "prompt_only"])
    parser.add_argument("--judge_reward_type", type=str, default="margin")
    parser.add_argument("--randomize_order", type=_parse_bool, default=True)
    parser.add_argument("--enable_thinking_prompt", type=_parse_bool, default=False)
    parser.add_argument("--output_dir", type=str, default="datasets/dpo_to_rupo")
    args = parser.parse_args()

    preprocess(
        dataset_name=args.dataset_name,
        policy_prompt_mode=args.policy_prompt_mode,
        judge_reward_type=args.judge_reward_type,
        randomize_order=args.randomize_order,
        enable_thinking_prompt=args.enable_thinking_prompt,
        output_dir=args.output_dir,
    )
