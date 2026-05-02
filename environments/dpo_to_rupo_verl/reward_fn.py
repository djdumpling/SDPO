"""Batch reward function for dpo_to_rupo inside verl's BatchRewardManager.

This module is the core integration between dpo_to_rupo's judge-based rubric
evaluation and verl/SDPO's training loop.  It runs all judge API calls
concurrently within a single batch to avoid sequential latency.

The function returns dicts with ``score`` and ``feedback`` keys so that SDPO's
self-distillation can use judge analysis as environment feedback.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from typing import Any

from openai import AsyncOpenAI

try:
    from .rubric_parser import extract_rubric_text, extract_score
except ImportError:
    # verl loads this file by raw file path (no package context), so relative
    # imports fail. Fall back to path-injected absolute import.
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from rubric_parser import extract_rubric_text, extract_score


# ---------------------------------------------------------------------------
# Judge prompt builders — inlined from dpo_to_rupo.prompts to avoid the
# verifiers import that prompts.py triggers at module level.
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_SYSTEM_PROMPT = """
You are an expert grader of creative-writing stories against scoring rubrics.

You will be given:
- a writing prompt,
- one candidate story written for that prompt,
- and a scoring rubric for evaluating that story on a 0-100 scale.

Your job is to grade the story against the rubric and return a single integer from 0 to 100, where 100 means the story satisfies the rubric extremely well."""


def _build_judge_prompt(prompt: str, response_text: str, rubric_text: str) -> str:
    return f"""You are grading one response against one rubric.

Here is the prompt:
<prompt>
{prompt}
</prompt>

Here is the response:
<response>
{response_text}
</response>

Here is the rubric:
<rubric>
{rubric_text}
</rubric>

Analyze how well the response satisfies the rubric for this prompt.
Then return:
<analysis>
Short justification grounded in the rubric and the response.
</analysis>
<score>
A single integer from 0 to 100, where 100 means the response satisfies the rubric extremely well.
</score>

Return only those XML tags.
"""


def _build_single_criterion_judge_prompt(
    prompt: str, response_text: str, criterion_index: int, criterion_name: str, criterion_description: str,
) -> str:
    return f"""You are grading one response on one rubric criterion.

Here is the prompt:
<prompt>
{prompt}
</prompt>

Here is the response:
<response>
{response_text}
</response>

Here is the criterion:
<criterion>
<index>{criterion_index}</index>
<name>{criterion_name}</name>
<description>{criterion_description}</description>
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


# ---------------------------------------------------------------------------
# Reward shaping — inlined from dpo_to_rupo.rewards to stay dependency-free.
# ---------------------------------------------------------------------------

_SIGMOID_MARGIN_SLOPE = math.log(0.95 / 0.05) / 10.0
_CLIPPED_MARGIN_DEADZONE = 5.0
_CLIPPED_MARGIN_CAP = 35.0
_CRITERIA_ABSOLUTE_DEADZONE = 5.0

_SCALAR_REWARD_TYPES = {"margin", "absolute", "sigmoid"}
_CRITERIA_REWARD_TYPES = {"criteria_margin", "criteria_absolute", "criteria_absolute_deadzone", "criteria_total_absolute"}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clipped_margin_reward(score_gap: float) -> float:
    absolute_gap = abs(score_gap)
    if absolute_gap <= _CLIPPED_MARGIN_DEADZONE:
        return 0.5
    clipped_gap = min(absolute_gap, _CLIPPED_MARGIN_CAP) - _CLIPPED_MARGIN_DEADZONE
    usable_range = _CLIPPED_MARGIN_CAP - _CLIPPED_MARGIN_DEADZONE
    signed_reward_offset = clipped_gap / (2.0 * usable_range)
    return 0.5 + signed_reward_offset if score_gap > 0.0 else 0.5 - signed_reward_offset


def _absolute_reward(score_gap: float) -> float:
    return 1.0 if score_gap > 0.0 else 0.5 if score_gap == 0.0 else 0.0


def _absolute_reward_with_deadzone(score_gap: float, deadzone: float) -> float:
    if abs(score_gap) <= deadzone:
        return 0.5
    return 1.0 if score_gap > 0.0 else 0.0


def _reward_from_scores(chosen_score: float, rejected_score: float, reward_type: str) -> float:
    gap = chosen_score - rejected_score
    if reward_type == "margin":
        return _clipped_margin_reward(gap)
    if reward_type == "absolute":
        return _absolute_reward(gap)
    if reward_type == "sigmoid":
        return _sigmoid(_SIGMOID_MARGIN_SLOPE * gap)
    raise ValueError(f"Unsupported scalar reward_type: {reward_type}")


def _reward_from_criterion_scores(
    chosen_scores: list[float], rejected_scores: list[float], criterion_weights: list[int], reward_type: str,
) -> float:
    if not chosen_scores or len(chosen_scores) != len(rejected_scores) or len(chosen_scores) != len(criterion_weights):
        return 0.0
    total_weight = sum(criterion_weights)
    if total_weight <= 0:
        return 0.0
    weighted_reward_sum = 0.0
    for c, r, w in zip(chosen_scores, rejected_scores, criterion_weights, strict=True):
        if w <= 0:
            return 0.0
        gap = c - r
        if reward_type == "criteria_margin":
            cr = _clipped_margin_reward(gap)
        elif reward_type == "criteria_absolute":
            cr = _absolute_reward(gap)
        elif reward_type == "criteria_absolute_deadzone":
            cr = _absolute_reward_with_deadzone(gap, _CRITERIA_ABSOLUTE_DEADZONE)
        else:
            raise ValueError(f"Unsupported criteria reward_type: {reward_type}")
        weighted_reward_sum += float(w) * cr
    return weighted_reward_sum / float(total_weight)


def _weighted_total(scores: list[float], weights: list[int]) -> float:
    if not scores or len(scores) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    return sum(float(w) / float(total_weight) * float(s) for s, w in zip(scores, weights, strict=True))


# ---------------------------------------------------------------------------
# Structured rubric parsing — import directly since it has no verifiers dep.
# Falls back to a bundled copy if dpo_to_rupo is not installed.
# ---------------------------------------------------------------------------

try:
    from dpo_to_rupo.structured_rubric import RubricCriterion, inspect_structured_rubric
except ImportError:
    import sys
    import os as _os
    # Try importing from the environments directory at the repo root
    _repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    _env_path = _os.path.join(_repo_root, "environments", "dpo_to_rupo")
    if _env_path not in sys.path:
        sys.path.insert(0, _env_path)
    from dpo_to_rupo.structured_rubric import RubricCriterion, inspect_structured_rubric


# ---------------------------------------------------------------------------
# Async judge execution
# ---------------------------------------------------------------------------

def _get_judge_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    timeout = float(os.environ.get("JUDGE_TIMEOUT", "900"))
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _get_judge_model() -> str:
    return os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


def _get_judge_max_tokens() -> int:
    return int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))


async def _run_judge_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    judge_prompt: str,
    model: str,
    max_tokens: int,
    system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
) -> tuple[str, float]:
    """Execute one judge call and return (raw_reply, score)."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": judge_prompt})

    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Judge returned empty content.")
    score = extract_score(content)
    if score is None:
        raise ValueError(f"Failed to extract score from judge reply: {content[:200]}")
    return content, score


# ---------------------------------------------------------------------------
# Per-sample scoring coroutine
# ---------------------------------------------------------------------------

async def _score_one_sample(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    max_tokens: int,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict[str, Any]:
    """Score one policy response by running the judge on chosen and rejected."""

    reward_type = extra_info.get("judge_reward_type", "margin")

    # 1. Parse rubric from policy output
    rubric_text = extract_rubric_text(solution_str)
    if rubric_text is None:
        return {
            "score": 0.0,
            "feedback": "Rubric extraction failed. Your response must contain <rubric>...</rubric> XML.",
            "rubric_parse_success": 0,
        }

    # 2. Parse ground truth
    try:
        payload = json.loads(ground_truth)
        prompt_text = str(payload["prompt"]).strip()
        chosen_text = str(payload["chosen"]).strip()
        rejected_text = str(payload["rejected"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {
            "score": 0.0,
            "feedback": f"Ground truth parse error: {e}",
            "rubric_parse_success": 1,
        }

    # 3. Route to scalar or structured scoring
    if reward_type in _CRITERIA_REWARD_TYPES:
        return await _score_criteria(
            client, semaphore, model, max_tokens,
            prompt_text, chosen_text, rejected_text, rubric_text, reward_type,
        )
    else:
        return await _score_scalar(
            client, semaphore, model, max_tokens,
            prompt_text, chosen_text, rejected_text, rubric_text, reward_type,
        )


async def _score_scalar(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    max_tokens: int,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
    rubric_text: str,
    reward_type: str,
) -> dict[str, Any]:
    """Score using holistic judge calls on chosen and rejected."""

    chosen_prompt = _build_judge_prompt(prompt_text, chosen_text, rubric_text)
    rejected_prompt = _build_judge_prompt(prompt_text, rejected_text, rubric_text)

    (chosen_reply, chosen_score), (rejected_reply, rejected_score) = await asyncio.gather(
        _run_judge_call(client, semaphore, chosen_prompt, model, max_tokens),
        _run_judge_call(client, semaphore, rejected_prompt, model, max_tokens),
    )

    reward = _reward_from_scores(chosen_score, rejected_score, reward_type)
    feedback = _build_feedback(chosen_score, rejected_score, reward, chosen_reply)
    return {
        "score": reward,
        "feedback": feedback,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "rubric_parse_success": 1,
    }


async def _score_criteria(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    max_tokens: int,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
    rubric_text: str,
    reward_type: str,
) -> dict[str, Any]:
    """Score using per-criterion judge calls."""

    # Parse structured rubric
    inspection = inspect_structured_rubric(rubric_text)
    if not inspection.valid:
        return {
            "score": 0.0,
            "feedback": (
                f"Structured rubric is invalid: criteria={inspection.criterion_count} "
                f"complete={inspection.complete_criterion_count} total_weight={inspection.total_weight}. "
                "The rubric must have 2+ criteria with positive integer weights summing to 100."
            ),
            "rubric_parse_success": 1,
        }

    criteria = list(inspection.criteria)

    # Score each criterion for both chosen and rejected concurrently
    async def _score_story_criteria(story_text: str) -> list[float]:
        tasks = [
            _run_judge_call(
                client, semaphore,
                _build_single_criterion_judge_prompt(prompt_text, story_text, c.index, c.name, c.description),
                model, max_tokens,
            )
            for c in criteria
        ]
        results = await asyncio.gather(*tasks)
        return [score for _, score in results]

    chosen_crit_scores, rejected_crit_scores = await asyncio.gather(
        _score_story_criteria(chosen_text),
        _score_story_criteria(rejected_text),
    )

    weights = [c.weight for c in criteria]

    if reward_type == "criteria_total_absolute":
        chosen_total = _weighted_total(chosen_crit_scores, weights)
        rejected_total = _weighted_total(rejected_crit_scores, weights)
        reward = _absolute_reward(chosen_total - rejected_total)
    else:
        reward = _reward_from_criterion_scores(chosen_crit_scores, rejected_crit_scores, weights, reward_type)

    chosen_total = _weighted_total(chosen_crit_scores, weights)
    rejected_total = _weighted_total(rejected_crit_scores, weights)
    feedback = _build_feedback(chosen_total, rejected_total, reward, judge_analysis=None)
    return {
        "score": reward,
        "feedback": feedback,
        "chosen_score": chosen_total,
        "rejected_score": rejected_total,
        "rubric_parse_success": 1,
    }


# ---------------------------------------------------------------------------
# Feedback builder for SDPO self-distillation
# ---------------------------------------------------------------------------

def _build_feedback(chosen_score: float, rejected_score: float, reward: float, judge_analysis: str | None) -> str:
    """Build a human-readable feedback string for SDPO's self-distillation."""
    if reward >= 0.7:
        msg = f"Good rubric. Chosen scored {chosen_score:.0f}, rejected scored {rejected_score:.0f} (reward={reward:.2f})."
    elif chosen_score < rejected_score:
        msg = (
            f"Your rubric scored the non-preferred response higher "
            f"(chosen={chosen_score:.0f}, rejected={rejected_score:.0f}, reward={reward:.2f}). "
            f"The rubric needs criteria that better capture what makes the preferred response stronger."
        )
    else:
        msg = (
            f"Your rubric did not discriminate well enough "
            f"(chosen={chosen_score:.0f}, rejected={rejected_score:.0f}, reward={reward:.2f}). "
            f"The rubric needs sharper criteria to separate the preferred from the non-preferred response."
        )
    if judge_analysis:
        msg += f" Judge analysis: {judge_analysis[:500]}"
    return msg


# ---------------------------------------------------------------------------
# Public batch scoring function — called by verl's BatchRewardManager
# ---------------------------------------------------------------------------

# Lazily initialized globals shared across calls within the same process.
_CLIENT: AsyncOpenAI | None = None
_SEMAPHORE: asyncio.Semaphore | None = None


def compute_score(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Score a batch of policy rubric outputs using async judge calls.

    Called by verl's ``BatchRewardManager.verify()`` with lists of decoded
    responses and metadata.  Returns a list of dicts, one per sample, each
    containing at minimum ``{"score": float, "feedback": str}``.
    """
    global _CLIENT, _SEMAPHORE

    if _CLIENT is None:
        _CLIENT = _get_judge_client()
    max_concurrent = int(os.environ.get("JUDGE_MAX_CONCURRENT", "64"))
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(max_concurrent)

    model = _get_judge_model()
    max_tokens = _get_judge_max_tokens()

    async def _score_batch() -> list[dict[str, Any]]:
        tasks = [
            _score_one_sample(_CLIENT, _SEMAPHORE, model, max_tokens, sol, gt, ei)
            for sol, gt, ei in zip(solution_strs, ground_truths, extra_infos, strict=True)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                scored.append({
                    "score": 0.0,
                    "feedback": f"Judge error: {result}",
                    "rubric_parse_success": 0,
                })
            else:
                scored.append(result)
        return scored

    # Run the async batch in a new event loop to avoid conflicts with
    # any existing loop in the Ray worker process.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # If there's already an event loop running (e.g. in Jupyter or nested),
        # use nest_asyncio to allow nested event loops.
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_score_batch())
    else:
        return asyncio.run(_score_batch())
