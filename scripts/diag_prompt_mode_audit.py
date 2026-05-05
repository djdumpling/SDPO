"""Phase 1 — train vs val prompt-mode audit (H8).

Checks whether the trained policy's training prompts and validation prompts use
the same `policy_prompt_mode` (pair vs prompt_only). If they differ, the
trained-vs-baseline comparison is structurally invalid.

Reads the launcher script + prepare script + preprocess_dataset.py to figure
out which mode is set for train/val. Also peeks at one rubric from each side to
confirm.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def grep_for(path: Path, patterns: list[str]) -> dict[str, list[str]]:
    """Return {pattern: [matched_line, ...]} for each line matching each pattern."""
    if not path.exists():
        return {p: [f"(file not found: {path})"] for p in patterns}
    text = path.read_text(errors="replace")
    out = {}
    for pat in patterns:
        out[pat] = [ln.strip() for ln in text.splitlines() if re.search(pat, ln)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--preprocess", required=True)
    ap.add_argument("--baseline-prep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    findings: dict = {}

    findings["launcher"] = grep_for(Path(args.launcher),
        [r"policy_prompt_mode", r"DATASET_NAME", r"TRAIN_DATA", r"EVAL_DATA",
         r"trainer\.val_before_train", r"VAL_KWARGS_N", r"ROLLOUT_N"])

    findings["preprocess"] = grep_for(Path(args.preprocess),
        [r"policy_prompt_mode", r"build_policy_prompt", r"is_eval", r"split"])

    findings["baseline_prep"] = grep_for(Path(args.baseline_prep),
        [r"policy_prompt_mode", r"DATASET_NAME"])

    # Concrete check: does the dataset prep flow set the same prompt_mode for train and val?
    # If preprocess_dataset.py routes by `is_eval` or `split`, that's the smoking gun.
    summary = {}
    pp_lines = "\n".join(findings["preprocess"].get("policy_prompt_mode", []))
    if "is_eval" in pp_lines or "split" in pp_lines:
        summary["differs_train_vs_val"] = "POSSIBLE — preprocess routes by is_eval/split; manual inspection required"
    else:
        summary["differs_train_vs_val"] = "UNLIKELY — preprocess does not branch by split"

    summary["recommendation"] = (
        "Open preprocess_dataset.py around the `policy_prompt_mode` line and confirm "
        "both train and val rows go through the same code path. The untrained baseline "
        "uses `policy_prompt_mode=prompt_only`; the trained policy uses `pair`. If val "
        "for the trained policy uses `pair` mode (sees both responses), and the "
        "untrained baseline uses `prompt_only`, the comparison is asymmetric."
    )

    out = {"findings": findings, "summary": summary}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
