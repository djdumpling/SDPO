"""Standalone XML extraction helpers for HA rubric rewards."""

from __future__ import annotations

import re


_RUBRIC_RE = re.compile(r"<rubric>(.*?)</rubric>", re.DOTALL)
_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL)
_SCORE_RE = re.compile(r"<score>(.*?)</score>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove optional <think>...</think> blocks that models may prepend."""
    return _THINK_RE.sub("", text)


def extract_rubric_text(response: str) -> str | None:
    """Extract the content of the <rubric>...</rubric> block."""
    cleaned = strip_think_tags(response)
    match = _RUBRIC_RE.search(cleaned)
    if match:
        text = match.group(1).strip()
        return text if text else None
    return None


def extract_analysis_text(response: str) -> str | None:
    """Extract the content of the <analysis>...</analysis> block."""
    cleaned = strip_think_tags(response)
    match = _ANALYSIS_RE.search(cleaned)
    if match:
        text = match.group(1).strip()
        return text if text else None
    return None


def extract_score(response: str) -> float | None:
    """Extract an integer score from a <score>...</score> block."""
    cleaned = strip_think_tags(response)
    match = _SCORE_RE.search(cleaned)
    if not match:
        return None
    try:
        score = int(match.group(1).strip())
    except ValueError:
        return None
    if score < 0 or score > 100:
        return None
    return float(score)
