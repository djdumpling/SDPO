"""Convert assistant-response preference pairs to verl's parquet format.

Usage:
    python -m environments.ha_rubric_verl.preprocess_dataset \
        --dataset_name sumuks/coval-ha \
        --policy_prompt_mode pair \
        --judge_reward_type margin \
        --output_dir datasets/ha_coval_margin

Produces train.parquet and test.parquet with verl's expected schema:
    data_source, prompt, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


SCALAR_REWARD_TYPES = {"margin", "absolute", "sigmoid"}

DEFAULT_PAIR_SYSTEM_PROMPT = """You are designing evaluation rubrics for assistant responses.

You will be given an original conversation and two candidate assistant responses.
Your task is to write a rubric that can be used to judge which assistant response is better for that conversation.

A strong rubric:
- focuses on the user's actual request and context,
- rewards helpfulness, correctness, instruction following, calibrated uncertainty, and clear reasoning,
- handles subjective or value-sensitive prompts with nuance, neutrality, and respect for user agency,
- penalizes hallucinations, overconfident claims, unsafe advice, evasiveness, irrelevant content, and unhelpful tone,
- is specific to this conversation rather than a generic checklist,
- does not mention Candidate A, Candidate B, "chosen", "rejected", or any hidden preference label.

Return exactly:
<analysis>
A concise explanation of what matters most for evaluating responses to this conversation.
</analysis>
<rubric>
The rubric criteria. Plain text bullets are fine.
</rubric>
"""

DEFAULT_PROMPT_ONLY_SYSTEM_PROMPT = """You are designing evaluation rubrics for assistant responses.

You will be given an original conversation. Write a prompt-specific rubric for judging assistant responses to that conversation.

Return exactly:
<analysis>
A concise explanation of what matters most for evaluating responses to this conversation.
</analysis>
<rubric>
The rubric criteria. Plain text bullets are fine.
</rubric>
"""

THINKING_SYSTEM_PROMPT_APPENDIX = """

Thinking-mode output contract
- Use the model's thinking channel for private scratch work when it is available.
- Close any `<think>` block before emitting the final XML.
- The final answer must contain exactly one closed `<analysis>...</analysis>` block followed by one closed `<rubric>...</rubric>` block.
- Keep the final `<analysis>` concise. Put detailed private reasoning in the thinking channel, not in the final XML.
- Always finish the closing `</rubric>` tag before the response ends.
"""


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
# Prompt and row mapping helpers
# ---------------------------------------------------------------------------

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


def _message_content(message: Any) -> str:
    if not isinstance(message, dict):
        return str(message)
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def _render_messages(value: Any, include_roles: bool) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return str(value).strip()

    rendered: list[str] = []
    for message in value:
        content = _message_content(message).strip()
        if not content:
            continue
        if include_roles and isinstance(message, dict):
            role = str(message.get("role", "message")).strip() or "message"
            rendered.append(f"{role}: {content}")
        else:
            rendered.append(content)
    return "\n\n".join(rendered).strip()


def _text_field(example: dict[str, Any], text_key: str, messages_key: str, include_roles: bool) -> str:
    structured_value = example.get(messages_key)
    if include_roles and isinstance(structured_value, list):
        rendered = _render_messages(structured_value, include_roles=True)
        if rendered:
            return rendered

    value = example.get(text_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _render_messages(structured_value, include_roles=include_roles)


def _get_system_prompt(policy_prompt_mode: str, custom_system_prompt: str | None, enable_thinking_prompt: bool) -> str:
    if custom_system_prompt:
        base = custom_system_prompt
    else:
        base = DEFAULT_PAIR_SYSTEM_PROMPT if policy_prompt_mode == "pair" else DEFAULT_PROMPT_ONLY_SYSTEM_PROMPT
    return base.rstrip() + THINKING_SYSTEM_PROMPT_APPENDIX if enable_thinking_prompt else base


def build_policy_prompt(prompt_text: str, candidate_a: str, candidate_b: str, policy_prompt_mode: str) -> str:
    if policy_prompt_mode == "prompt_only":
        return f"""Original conversation:
<conversation>
{prompt_text}
</conversation>

Write an evaluation rubric for assistant responses to this conversation."""

    return f"""Original conversation:
<conversation>
{prompt_text}
</conversation>

Candidate assistant response A:
<response_a>
{candidate_a}
</response_a>

Candidate assistant response B:
<response_b>
{candidate_b}
</response_b>

Write an evaluation rubric that would help a judge determine which candidate response is better for this conversation."""


def make_map_fn(
    split: str,
    policy_prompt_mode: str,
    judge_reward_type: str,
    randomize_order: bool,
    system_prompt: str,
):
    """Return a row mapping function for datasets.map()."""

    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        prompt_text = _text_field(example, "prompt_text", "prompt", include_roles=True)
        chosen_text = _text_field(example, "chosen_text", "chosen", include_roles=False)
        rejected_text = _text_field(example, "rejected_text", "rejected", include_roles=False)

        # Randomize candidate A/B order in pair mode to prevent position bias.
        swap = policy_prompt_mode == "pair" and randomize_order and random.random() < 0.5
        candidate_a = rejected_text if swap else chosen_text
        candidate_b = chosen_text if swap else rejected_text

        user_content = build_policy_prompt(prompt_text, candidate_a, candidate_b, policy_prompt_mode)

        # The reward function uses this to recover the preference pair for judge scoring.
        ground_truth = json.dumps(
            {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text},
            separators=(",", ":"),
        )

        return {
            "data_source": "ha_rubric",
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
                "source_prompt_id": str(example.get("prompt_id", "")),
                "source_annotator_id": str(example.get("annotator_id", "")),
                "source_preference_type": str(example.get("preference_type", "")),
                "source_ranking": str(example.get("ranking", "")),
                "source_rank_gap": str(example.get("rank_gap", "")),
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
    output_dir: str = "datasets/ha_coval",
) -> None:
    if judge_reward_type not in SCALAR_REWARD_TYPES:
        raise ValueError(f"HA reward supports scalar reward types only: {sorted(SCALAR_REWARD_TYPES)}")

    os.makedirs(output_dir, exist_ok=True)
    system_prompt = _get_system_prompt(policy_prompt_mode, custom_system_prompt, enable_thinking_prompt)

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
    parser = argparse.ArgumentParser(description="Convert HA preference datasets to verl parquet for SDPO rubric training.")
    parser.add_argument("--dataset_name", type=str, default="sumuks/coval-ha", help="HuggingFace dataset name.")
    parser.add_argument("--policy_prompt_mode", type=str, default="pair", choices=["pair", "prompt_only"])
    parser.add_argument("--judge_reward_type", type=str, default="margin", choices=sorted(SCALAR_REWARD_TYPES))
    parser.add_argument("--randomize_order", type=_parse_bool, default=True)
    parser.add_argument("--enable_thinking_prompt", type=_parse_bool, default=False)
    parser.add_argument("--custom_system_prompt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="datasets/ha_coval")
    args = parser.parse_args()

    preprocess(
        dataset_name=args.dataset_name,
        policy_prompt_mode=args.policy_prompt_mode,
        judge_reward_type=args.judge_reward_type,
        randomize_order=args.randomize_order,
        custom_system_prompt=args.custom_system_prompt,
        enable_thinking_prompt=args.enable_thinking_prompt,
        output_dir=args.output_dir,
    )
