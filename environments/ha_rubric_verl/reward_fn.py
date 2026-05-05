"""Batch reward function for assistant-response preference rubric training.

The policy generates a rubric. This reward function asks an OpenAI-compatible
judge to score the human-preferred and non-preferred assistant responses against
that rubric, then rewards rubrics that score the preferred response higher.
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
    # verl loads this file by raw file path, so package-relative imports may fail.
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from rubric_parser import extract_rubric_text, extract_score


DEFAULT_JUDGE_SYSTEM_PROMPT = """
You are an expert grader of assistant responses against evaluation rubrics.

You will be given:
- an original conversation,
- one candidate assistant response,
- and a scoring rubric for evaluating that response on a 0-100 scale.

Your job is to grade the response against the rubric and return a single integer from 0 to 100, where 100 means the response satisfies the rubric extremely well.
"""

_SIGMOID_MARGIN_SLOPE = math.log(0.95 / 0.05) / 10.0
_CLIPPED_MARGIN_DEADZONE = 5.0
_CLIPPED_MARGIN_CAP = 35.0
_SCALAR_REWARD_TYPES = {"margin", "absolute", "sigmoid"}


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


def _reward_from_scores(chosen_score: float, rejected_score: float, reward_type: str) -> float:
    gap = chosen_score - rejected_score
    if reward_type == "margin":
        return _clipped_margin_reward(gap)
    if reward_type == "absolute":
        return _absolute_reward(gap)
    if reward_type == "sigmoid":
        return _sigmoid(_SIGMOID_MARGIN_SLOPE * gap)
    raise ValueError(f"Unsupported scalar reward_type: {reward_type}")


def _build_judge_prompt(prompt: str, response_text: str, rubric_text: str) -> str:
    return f"""You are grading one assistant response against one rubric.

Here is the original conversation:
<conversation>
{prompt}
</conversation>

Here is the candidate assistant response:
<response>
{response_text}
</response>

Here is the rubric:
<rubric>
{rubric_text}
</rubric>

Analyze how well the response satisfies the rubric for this conversation.
Then return:
<analysis>
Short justification grounded in the rubric, the conversation, and the response.
</analysis>
<score>
A single integer from 0 to 100, where 100 means the response satisfies the rubric extremely well.
</score>

Return only those XML tags.
"""


def _get_judge_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    timeout = float(os.environ.get("JUDGE_TIMEOUT", "900"))
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _get_judge_model() -> str:
    return os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


def _get_judge_max_tokens() -> int:
    return int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_judge_extra_body() -> dict[str, Any] | None:
    enable_thinking = _env_bool("JUDGE_ENABLE_THINKING", None)
    if enable_thinking is None:
        return None
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


def _format_diagnostics(solution_str: str, rubric_text: str | None = None) -> dict[str, Any]:
    return {
        "rubric_open_seen": int("<rubric>" in solution_str),
        "rubric_close_seen": int("</rubric>" in solution_str),
        "analysis_open_seen": int("<analysis>" in solution_str),
        "analysis_close_seen": int("</analysis>" in solution_str),
        "think_open_seen": int("<think>" in solution_str),
        "think_close_seen": int("</think>" in solution_str),
        "response_char_count": len(solution_str),
        "rubric_char_count": len(rubric_text or ""),
    }


def _format_progress_reward(solution_str: str) -> float:
    cap = float(os.environ.get("RUBRIC_FORMAT_REWARD_MAX", "0.0"))
    if cap <= 0.0:
        return 0.0
    progress = 0.0
    if "<think>" in solution_str and "</think>" in solution_str:
        progress += 0.2
    if "<analysis>" in solution_str:
        progress += 0.15
    if "<analysis>" in solution_str and "</analysis>" in solution_str:
        progress += 0.15
    if "<rubric>" in solution_str:
        progress += 0.25
    if "<rubric>" in solution_str and "</rubric>" in solution_str:
        progress += 0.25
    return cap * min(progress, 1.0)


def _missing_judge_scores() -> dict[str, float]:
    return {"chosen_score": float("nan"), "rejected_score": float("nan")}


async def _run_judge_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    judge_prompt: str,
    model: str,
    max_tokens: int,
    system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
) -> tuple[str, float]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": judge_prompt})

    extra_body = _get_judge_extra_body()
    request_kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    async with semaphore:
        response = await client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Judge returned empty content.")
    score = extract_score(content)
    if score is None:
        raise ValueError(f"Failed to extract score from judge reply: {content[:200]}")
    return content, score


async def _score_one_sample(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    max_tokens: int,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict[str, Any]:
    reward_type = extra_info.get("judge_reward_type", "margin")
    if reward_type not in _SCALAR_REWARD_TYPES:
        return {
            **_format_diagnostics(solution_str),
            **_missing_judge_scores(),
            "score": 0.0,
            "feedback": f"Unsupported HA reward type: {reward_type}. Use one of {sorted(_SCALAR_REWARD_TYPES)}.",
            "rubric_parse_success": 0,
            "format_reward": 0.0,
            "judge_called": 0,
            "judge_error": 0,
            "judge_score_parse_success": 0,
        }

    diagnostics = _format_diagnostics(solution_str)
    rubric_text = extract_rubric_text(solution_str)
    if rubric_text is None:
        format_reward = _format_progress_reward(solution_str)
        return {
            **diagnostics,
            **_missing_judge_scores(),
            "score": format_reward,
            "feedback": "Rubric extraction failed. Your response must contain <rubric>...</rubric> XML.",
            "rubric_parse_success": 0,
            "format_reward": format_reward,
            "judge_called": 0,
            "judge_error": 0,
            "judge_score_parse_success": 0,
        }
    diagnostics = _format_diagnostics(solution_str, rubric_text)

    try:
        payload = json.loads(ground_truth)
        prompt_text = str(payload["prompt"]).strip()
        chosen_text = str(payload["chosen"]).strip()
        rejected_text = str(payload["rejected"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {
            **diagnostics,
            **_missing_judge_scores(),
            "score": 0.0,
            "feedback": f"Ground truth parse error: {e}",
            "rubric_parse_success": 1,
            "format_reward": 0.0,
            "judge_called": 0,
            "judge_error": 0,
            "judge_score_parse_success": 0,
        }

    try:
        result = await _score_scalar(
            client, semaphore, model, max_tokens,
            prompt_text, chosen_text, rejected_text, rubric_text, reward_type,
        )
    except Exception as e:
        return {
            **diagnostics,
            **_missing_judge_scores(),
            "score": 0.0,
            "feedback": f"Judge error: {e}",
            "rubric_parse_success": 1,
            "format_reward": 0.0,
            "judge_called": 1,
            "judge_error": 1,
            "judge_score_parse_success": 0,
        }

    return {
        **diagnostics,
        "format_reward": 0.0,
        "judge_called": 1,
        "judge_error": 0,
        "judge_score_parse_success": 1,
        **result,
    }


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
    chosen_prompt = _build_judge_prompt(prompt_text, chosen_text, rubric_text)
    rejected_prompt = _build_judge_prompt(prompt_text, rejected_text, rubric_text)

    (chosen_reply, chosen_score), (_rejected_reply, rejected_score) = await asyncio.gather(
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


def _build_feedback(chosen_score: float, rejected_score: float, reward: float, judge_analysis: str | None) -> str:
    if reward >= 0.7:
        msg = f"Good rubric. Preferred response scored {chosen_score:.0f}, non-preferred scored {rejected_score:.0f} (reward={reward:.2f})."
    elif chosen_score < rejected_score:
        msg = (
            f"Your rubric scored the non-preferred response higher "
            f"(preferred={chosen_score:.0f}, non-preferred={rejected_score:.0f}, reward={reward:.2f}). "
            f"The rubric needs criteria that better capture what makes the preferred assistant response stronger."
        )
    else:
        msg = (
            f"Your rubric did not discriminate well enough "
            f"(preferred={chosen_score:.0f}, non-preferred={rejected_score:.0f}, reward={reward:.2f}). "
            f"The rubric needs sharper criteria to separate the preferred response from the non-preferred response."
        )
    if judge_analysis:
        msg += f" Judge analysis: {judge_analysis[:500]}"
    return msg


_CLIENT: AsyncOpenAI | None = None
_SEMAPHORE: asyncio.Semaphore | None = None


def compute_score(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Score a batch of generated rubrics using async judge calls."""
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
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                scored.append({
                    **_format_diagnostics(solution_strs[i]),
                    **_missing_judge_scores(),
                    "score": 0.0,
                    "feedback": f"Judge error: {result}",
                    "rubric_parse_success": 0,
                    "format_reward": 0.0,
                    "judge_called": 0,
                    "judge_error": 1,
                    "judge_score_parse_success": 0,
                })
            else:
                scored.append(result)
        return scored

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(_score_batch())
    return asyncio.run(_score_batch())
