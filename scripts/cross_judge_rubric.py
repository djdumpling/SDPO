"""Cross-judge eval for the SDPO RuPO pipeline.

Re-scores policy rubric outputs with an alternative judge (e.g., GPT-5 via Prime
Intellect inference) so we can report cross-judge preference accuracy alongside
the original Qwen3-8B self-judge number. Reuses the same per-criterion judge
prompt and rubric parsing as the SDPO env so cross-judge results are
apples-to-apples with the in-loop reward.

Two input sources:
  - --source jsonl: read verl validation_data_dir output (one JSONL per val
    step, fields: input/output/gts/score). Best when run after a fresh training
    run with trainer.validation_data_dir set.
  - --source wandb: pull val/generations artifacts from a wandb run and match
    each row back to the val parquet for ground truth (chosen vs rejected).
    Useful when you want to re-judge a prior run that did not dump to disk.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Path bootstrap so we can import the SDPO env helpers without installing the
# packages. These are the canonical sources for rubric parsing + criterion
# scoring; vendoring would be one more thing to keep in sync.
# -----------------------------------------------------------------------------

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SDPO_ROOT = _HERE.parent
_REPO_ROOT = _SDPO_ROOT.parent
sys.path.insert(0, str(_SDPO_ROOT / "environments" / "dpo_to_rupo_verl"))
sys.path.insert(0, str(_REPO_ROOT / "environments" / "dpo_to_rupo"))

import argparse
import asyncio
import hashlib
import json
import re
import tempfile
import time
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from dpo_to_rupo.structured_rubric import inspect_structured_rubric
from rubric_parser import extract_rubric_text, extract_score


# -----------------------------------------------------------------------------
# Judge prompts. These match SDPO env's reward_fn._build_single_criterion_judge_prompt
# and DEFAULT_JUDGE_SYSTEM_PROMPT exactly so cross-judge scores are comparable
# to the in-loop Qwen3-8B reward by procedure as well as outcome.
# -----------------------------------------------------------------------------

DEFAULT_JUDGE_SYSTEM_PROMPT = """
You are an expert grader of creative-writing stories against scoring rubrics.

You will be given:
- a writing prompt,
- one candidate story written for that prompt,
- and a scoring rubric for evaluating that story on a 0-100 scale.

Your job is to grade the story against the rubric and return a single integer from 0 to 100, where 100 means the story satisfies the rubric extremely well."""


def build_single_criterion_judge_prompt(
    prompt_text: str, story_text: str, criterion_index: int, criterion_name: str, criterion_description: str,
) -> str:
    """Build the per-criterion grading prompt. Mirrors the SDPO env reward_fn."""
    return f"""You are grading one response on one rubric criterion.

Here is the prompt:
<prompt>
{prompt_text}
</prompt>

Here is the response:
<response>
{story_text}
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


# -----------------------------------------------------------------------------
# Reward shaping helpers. Inlined from the SDPO env so the cross-judge path has
# no verl dependency. Only criteria_total_absolute is needed for the current
# winning configuration; other reward types are left out to keep the script
# small and readable.
# -----------------------------------------------------------------------------

def absolute_reward(score_gap: float) -> float:
    """Sign of (chosen - rejected). 1.0 chosen wins, 0.5 tie, 0.0 rejected wins."""
    if score_gap > 0.0: return 1.0
    if score_gap == 0.0: return 0.5
    return 0.0


def weighted_total(scores: list[float], weights: list[int]) -> float:
    """Weighted average of per-criterion scores using the criterion weights."""
    if not scores or len(scores) != len(weights): return 0.0
    total_weight = sum(weights)
    if total_weight <= 0: return 0.0
    return sum((float(w) / float(total_weight)) * float(s) for s, w in zip(scores, weights, strict=True))


# -----------------------------------------------------------------------------
# One judge call. Wraps the AsyncOpenAI request with a per-call timeout and
# converts API exceptions into a typed error string instead of bubbling so
# one bad call doesn't poison the whole asyncio.gather batch.
# -----------------------------------------------------------------------------

async def run_one_judge_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    judge_prompt: str,
    model: str,
    max_completion_tokens: int,
    system_prompt: str,
    timeout_seconds: float,
    temperature: float | None = None,
) -> tuple[str | None, float | None, str | None]:
    """Return (raw_reply, parsed_score, error_kind)."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": judge_prompt}]
    extra: dict[str, Any] = {}
    if temperature is not None: extra["temperature"] = temperature
    async with semaphore:
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await client.chat.completions.create(
                    model=model, messages=messages, max_completion_tokens=max_completion_tokens,
                    **extra,
                )
        except (asyncio.TimeoutError, APITimeoutError):
            return None, None, "timeout"
        except RateLimitError:
            return None, None, "rate_limit"
        except APIError as e:
            return None, None, f"api_error_{getattr(e, 'status_code', 'unknown')}"
        except Exception as e:
            return None, None, f"other_{type(e).__name__}"
    content = response.choices[0].message.content if response.choices else None
    if not content: return None, None, "empty_content"
    parsed = extract_score(content)
    if parsed is None: return content, None, "score_parse_failed"
    return content, parsed, None


# -----------------------------------------------------------------------------
# Per-example scoring. Score chosen and rejected concurrently across criteria,
# matching the SDPO env's _score_criteria batching.
# -----------------------------------------------------------------------------

async def cross_judge_one_example(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    max_completion_tokens: int,
    system_prompt: str,
    timeout_seconds: float,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
    criteria: list[Any],
    temperature: float | None = None,
) -> dict[str, Any]:
    """Score one (prompt, chosen, rejected, rubric) tuple. Returns scores + reward."""
    weights = [c.weight for c in criteria]

    async def _score_story(story_text: str) -> list[tuple[str | None, float | None, str | None]]:
        return await asyncio.gather(*[
            run_one_judge_call(
                client, semaphore,
                build_single_criterion_judge_prompt(prompt_text, story_text, c.index, c.name, c.description),
                judge_model, max_completion_tokens, system_prompt, timeout_seconds,
                temperature=temperature,
            )
            for c in criteria
        ])

    chosen_results, rejected_results = await asyncio.gather(_score_story(chosen_text), _score_story(rejected_text))
    chosen_scores = [r[1] for r in chosen_results]
    rejected_scores = [r[1] for r in rejected_results]
    errors = [r[2] for r in chosen_results + rejected_results if r[2] is not None]

    if any(s is None for s in chosen_scores) or any(s is None for s in rejected_scores):
        return {"judge_called": True, "judge_error": True, "errors": errors[:3],
                "chosen_scores": chosen_scores, "rejected_scores": rejected_scores}

    chosen_total = weighted_total(chosen_scores, weights)
    rejected_total = weighted_total(rejected_scores, weights)
    reward = absolute_reward(chosen_total - rejected_total)
    return {"judge_called": True, "judge_error": False, "errors": [],
            "chosen_scores": chosen_scores, "rejected_scores": rejected_scores,
            "chosen_total": chosen_total, "rejected_total": rejected_total, "reward": reward}


# -----------------------------------------------------------------------------
# Source: verl validation_data_dir JSONL.
# Each line has input/output/gts/score plus reward_extra_infos. ground truth is
# a JSON-encoded {prompt, chosen, rejected} blob — the same structure the
# in-loop reward function consumes, so we can reuse it directly.
# -----------------------------------------------------------------------------

def load_examples_from_jsonl_file(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read one verl validation JSONL into our per-example dict format."""
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(jsonl_path.read_text().splitlines()):
        if not line.strip(): continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        gts_raw = d.get("gts")
        if not isinstance(gts_raw, str): continue
        try:
            gt = json.loads(gts_raw)
        except json.JSONDecodeError:
            continue
        rows.append({
            "policy_output": d.get("output", ""),
            "original_score": float(d["score"]) if d.get("score") is not None else float("nan"),
            "prompt": str(gt.get("prompt", "")).strip(),
            "chosen": str(gt.get("chosen", "")).strip(),
            "rejected": str(gt.get("rejected", "")).strip(),
            "step": d.get("step"),
            "src_file": jsonl_path.name,
            "src_line": line_no,
        })
    return rows


def load_examples_from_jsonl_path(path: Path, jsonl_step: str) -> list[dict[str, Any]]:
    """Resolve --jsonl into a list of files, then concatenate examples.

    --jsonl-step controls how a directory of {step}.jsonl files is reduced:
      - "latest": only the highest-numbered step (typical "evaluate the trained policy")
      - "all": concatenate all steps (use if you want the full trajectory)
      - integer string: that specific step file
    """
    if path.is_file(): return load_examples_from_jsonl_file(path)
    if not path.is_dir():
        raise SystemExit(f"--jsonl path does not exist: {path}")
    step_files = sorted(path.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not step_files:
        raise SystemExit(f"No *.jsonl files found in {path}")
    if jsonl_step == "latest":
        chosen_files = [step_files[-1]]
    elif jsonl_step == "all":
        chosen_files = step_files
    else:
        target = path / f"{jsonl_step}.jsonl"
        if not target.exists():
            raise SystemExit(f"Requested step JSONL not found: {target}")
        chosen_files = [target]
    print(f"  reading {len(chosen_files)} JSONL file(s) from {path}: {[f.name for f in chosen_files]}")
    rows: list[dict[str, Any]] = []
    for f in chosen_files: rows.extend(load_examples_from_jsonl_file(f))
    return rows


# -----------------------------------------------------------------------------
# Source: wandb val/generations artifact.
# The table structure is one row per validation step, with columns laid out as
# step, input_1, output_1, score_1, ..., input_K, output_K, score_K. To recover
# (chosen, rejected) we look up the parquet row whose user message matches the
# wandb input_N — the wandb log doesn't preserve candidate_a_is_chosen.
# -----------------------------------------------------------------------------

# Verl's wandb generations logger calls tokenizer.batch_decode(skip_special_tokens=True),
# so Qwen3's <|im_start|>/<|im_end|> markers are stripped. The remaining format is:
#   system\n{system_prompt}\nuser\n{user_message}\nassistant\n<think>...
# We extract everything between "\nuser\n" and "\nassistant". Note that
# detokenization can drop trailing whitespace that the parquet preserves;
# _user_msg_fingerprint normalizes that away.
_USER_RE = re.compile(r"\nuser\n(.*?)\nassistant\b", re.DOTALL)
_CAND_A_RE = re.compile(r"<candidate_a>(.*?)</candidate_a>", re.DOTALL)
_CAND_B_RE = re.compile(r"<candidate_b>(.*?)</candidate_b>", re.DOTALL)
_PROMPT_RE = re.compile(r"Here is the prompt:\s*\n(.*?)\n+<candidate_a>", re.DOTALL)


def _user_message(chat_input: str) -> str | None:
    """Extract the user-role content from a wandb-decoded chat input string."""
    m = _USER_RE.search(chat_input)
    return m.group(1) if m else None


def _user_msg_fingerprint(text: str) -> str:
    """Stable lookup key for matching wandb inputs to parquet rows.

    Strips trailing whitespace because verl's tokenizer.batch_decode drops the
    trailing newline that the parquet user-message text retains; without
    normalizing we'd miss every match by a single character.
    """
    return hashlib.sha1(text.rstrip().encode("utf-8", errors="replace")).hexdigest()


def _build_parquet_lookup(parquet_path: Path) -> dict[str, dict[str, Any]]:
    """Build a {fingerprint(user_message) -> ground_truth_row} index over the val parquet."""
    import pyarrow.parquet as pq
    df = pq.read_table(str(parquet_path)).to_pandas()
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        user_msg = next((m["content"] for m in row["prompt"] if m["role"] == "user"), None)
        if not user_msg: continue
        ca_m = _CAND_A_RE.search(user_msg)
        cb_m = _CAND_B_RE.search(user_msg)
        if not ca_m or not cb_m: continue
        candidate_a = ca_m.group(1).strip()
        candidate_b = cb_m.group(1).strip()
        extra = row.get("extra_info") or {}
        candidate_a_is_chosen = bool(extra.get("candidate_a_is_chosen"))
        chosen, rejected = (candidate_a, candidate_b) if candidate_a_is_chosen else (candidate_b, candidate_a)
        wp_m = _PROMPT_RE.search(user_msg)
        prompt_text = wp_m.group(1).strip() if wp_m else ""
        lookup[_user_msg_fingerprint(user_msg)] = {
            "prompt": prompt_text, "chosen": chosen, "rejected": rejected,
            "candidate_a_is_chosen": candidate_a_is_chosen, "index": extra.get("index"),
        }
    print(f"  parquet lookup: {len(lookup)} unique val prompts indexed.")
    return lookup


def load_examples_from_wandb(
    wandb_run_path: str, parquet_path: Path, val_step_filter: int | None,
) -> list[dict[str, Any]]:
    """Pull val/generations from a wandb run and resolve chosen/rejected via parquet lookup."""
    import wandb
    api = wandb.Api()
    run = api.run(wandb_run_path)
    arts = [a for a in run.logged_artifacts() if "valgenerations" in a.name]
    if not arts:
        raise SystemExit(f"No val/generations artifacts on wandb run {wandb_run_path}")
    arts.sort(key=lambda a: int(a.name.split(":v")[1]))
    print(f"  wandb run: {wandb_run_path} — {len(arts)} val artifact version(s)")
    parquet_lookup = _build_parquet_lookup(parquet_path)

    rows: list[dict[str, Any]] = []
    n_no_match = 0
    for art in arts:
        with tempfile.TemporaryDirectory() as td:
            art.download(root=td)
            tbl_path = next(Path(td).rglob("generations.table.json"), None)
            if tbl_path is None: continue
            data = json.loads(tbl_path.read_text())
            cols = data.get("columns", [])
            for table_row in data.get("data", []):
                step = table_row[0] if cols and cols[0] == "step" else None
                if val_step_filter is not None and step != val_step_filter: continue
                # Walk every (input_N, output_N, score_N) triple
                triple_idx = 1
                while True:
                    in_key = f"input_{triple_idx}"
                    out_key = f"output_{triple_idx}"
                    sc_key = f"score_{triple_idx}"
                    if in_key not in cols or out_key not in cols or sc_key not in cols: break
                    in_val = table_row[cols.index(in_key)]
                    out_val = table_row[cols.index(out_key)]
                    sc_val = table_row[cols.index(sc_key)]
                    triple_idx += 1
                    if not isinstance(in_val, str) or not isinstance(out_val, str): continue
                    user_msg = _user_message(in_val)
                    if user_msg is None:
                        n_no_match += 1
                        continue
                    gt = parquet_lookup.get(_user_msg_fingerprint(user_msg))
                    if gt is None:
                        n_no_match += 1
                        continue
                    rows.append({
                        "policy_output": out_val,
                        "original_score": float(sc_val) if sc_val is not None else float("nan"),
                        "prompt": gt["prompt"], "chosen": gt["chosen"], "rejected": gt["rejected"],
                        "step": step, "src_file": art.name, "src_idx": triple_idx - 1,
                    })
    if n_no_match: print(f"  warning: {n_no_match} wandb rows could not be matched to parquet ground truth (skipped)")
    return rows


# -----------------------------------------------------------------------------
# Main async loop. Per-example pipeline: parse rubric → score chosen+rejected
# against each criterion → aggregate → record. Concurrency is controlled by
# one semaphore shared across all judge calls so we never exceed --max-concurrent
# in-flight requests regardless of how many examples are pending.
# -----------------------------------------------------------------------------

async def run_cross_judge(args: argparse.Namespace) -> dict[str, Any]:
    """Top-level async driver. Loads examples, runs cross-judge, returns summary dict."""
    if args.source == "jsonl":
        examples = load_examples_from_jsonl_path(Path(args.jsonl), args.jsonl_step)
    elif args.source == "wandb":
        if not args.wandb_run or not args.parquet:
            raise SystemExit("--source wandb requires --wandb-run and --parquet")
        examples = load_examples_from_wandb(args.wandb_run, Path(args.parquet), args.wandb_step)
    else:
        raise SystemExit(f"Unknown --source: {args.source}")
    if args.limit is not None and args.limit > 0: examples = examples[: args.limit]
    print(f"  loaded {len(examples)} example(s) to cross-judge with {args.judge_model}")

    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key: raise SystemExit(f"Missing API key in env var {args.judge_api_key_env}")
    client = AsyncOpenAI(api_key=api_key, base_url=args.judge_base_url, timeout=args.judge_timeout)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    n_total = len(examples)
    counters = {"done": 0, "rubric_parse_fail": 0, "rubric_invalid": 0, "judge_error": 0}
    rewards_crossjudge: list[float] = []
    rewards_original: list[float] = []
    per_row: list[dict[str, Any]] = []
    started = time.time()

    async def _process(idx: int, ex: dict[str, Any]) -> dict[str, Any]:
        rubric_text = extract_rubric_text(ex["policy_output"])
        if rubric_text is None:
            counters["rubric_parse_fail"] += 1
            return {"idx": idx, **ex, "rubric_parse_success": False, "judge_called": False, "reward": 0.0}
        inspection = inspect_structured_rubric(rubric_text)
        if not inspection.valid:
            counters["rubric_invalid"] += 1
            return {"idx": idx, **ex, "rubric_parse_success": True, "valid_rubric": False,
                    "judge_called": False, "reward": 0.0,
                    "rubric_diagnostics": {"criterion_count": inspection.criterion_count,
                                            "complete_criterion_count": inspection.complete_criterion_count,
                                            "total_weight": inspection.total_weight}}
        result = await cross_judge_one_example(
            client, semaphore, args.judge_model, args.judge_max_tokens,
            DEFAULT_JUDGE_SYSTEM_PROMPT, args.judge_timeout,
            ex["prompt"], ex["chosen"], ex["rejected"], list(inspection.criteria),
            temperature=args.temperature,
        )
        if result.get("judge_error"): counters["judge_error"] += 1
        return {"idx": idx, **ex, "rubric_parse_success": True, "valid_rubric": True, **result}

    tasks = [asyncio.create_task(_process(i, ex)) for i, ex in enumerate(examples)]
    for fut in asyncio.as_completed(tasks):
        row = await fut
        per_row.append(row)
        counters["done"] += 1
        if row.get("judge_called") and not row.get("judge_error") and "reward" in row:
            rewards_crossjudge.append(float(row["reward"]))
        orig = row.get("original_score")
        if orig is not None and not (orig != orig):  # skip NaN
            rewards_original.append(float(orig))
        if counters["done"] % 25 == 0 or counters["done"] == n_total:
            elapsed = time.time() - started
            rate = counters["done"] / max(elapsed, 1e-6)
            print(f"  progress: {counters['done']}/{n_total}  "
                  f"rubric_parse_fail={counters['rubric_parse_fail']}  "
                  f"rubric_invalid={counters['rubric_invalid']}  "
                  f"judge_err={counters['judge_error']}  "
                  f"rate={rate:.2f} ex/s  elapsed={elapsed:.0f}s")

    per_row.sort(key=lambda r: r["idx"])

    n_eval = len(rewards_crossjudge)
    summary = {
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "source": args.source,
        "wandb_run": args.wandb_run if args.source == "wandb" else None,
        "jsonl_path": args.jsonl if args.source == "jsonl" else None,
        "jsonl_step": args.jsonl_step if args.source == "jsonl" else None,
        "wandb_step_filter": args.wandb_step if args.source == "wandb" else None,
        "limit": args.limit,
        "n_total": n_total,
        "n_rubric_parse_fail": counters["rubric_parse_fail"],
        "n_rubric_invalid": counters["rubric_invalid"],
        "n_judge_error": counters["judge_error"],
        "n_evaluated": n_eval,
        "mean_crossjudge_reward": (sum(rewards_crossjudge) / n_eval) if n_eval else float("nan"),
        "mean_original_reward": (sum(rewards_original) / len(rewards_original)) if rewards_original else float("nan"),
        "n_original_with_score": len(rewards_original),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return {"summary": summary, "rows": per_row}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define the CLI for the cross-judge runner."""
    p = argparse.ArgumentParser(description="Re-judge SDPO RuPO policy outputs with an alternative judge.")
    p.add_argument("--source", choices=["jsonl", "wandb"], required=True,
                   help="Where to read policy outputs from.")
    p.add_argument("--jsonl", help="Path to verl validation JSONL file or directory of {step}.jsonl files.")
    p.add_argument("--jsonl-step", default="latest",
                   help='Reduce a directory of step JSONLs: "latest" (default), "all", or a step number string.')
    p.add_argument("--wandb-run", help="Wandb run path entity/project/run_id (used with --source wandb).")
    p.add_argument("--parquet",
                   default=str(_SDPO_ROOT / "datasets" / "dpo_to_rupo_litbench_ha_with_baseline_criteria_total_absolute" / "test.parquet"),
                   help="Val parquet for ground-truth lookup when reading from wandb.")
    p.add_argument("--wandb-step", type=int, default=None,
                   help="If set, restrict wandb tables to this val step only (e.g., 100).")
    p.add_argument("--judge-model", default="openai/gpt-5", help="Cross-judge model id.")
    p.add_argument("--judge-base-url", default="https://api.pinference.ai/api/v1",
                   help="OpenAI-compatible base URL. Default points at Prime Intellect inference.")
    p.add_argument("--judge-api-key-env", default="PRIME_API_KEY",
                   help="Env var holding the judge API key.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature for judge calls. Default: server default. "
                        "Set to 0.7 for OpenRubrics-style stochastic judge ensembling.")
    p.add_argument("--judge-max-tokens", type=int, default=4096,
                   help="max_completion_tokens per judge call (covers reasoning tokens for GPT-5).")
    p.add_argument("--judge-timeout", type=float, default=300.0,
                   help="Per-call timeout in seconds.")
    p.add_argument("--max-concurrent", type=int, default=32,
                   help="Maximum in-flight judge calls. Lower if you hit rate limits.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on the number of examples to cross-judge.")
    p.add_argument("--output", required=True, help="JSON output path for {summary, rows}.")
    return p.parse_args()


def main() -> None:
    """Entry point. Runs the async driver and writes the JSON output."""
    args = parse_args()
    print(f"cross_judge_rubric: source={args.source}  judge={args.judge_model}  base_url={args.judge_base_url}")
    result = asyncio.run(run_cross_judge(args))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n=== summary ===")
    for k, v in result["summary"].items(): print(f"  {k}: {v}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__": main()
