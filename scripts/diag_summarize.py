"""Phase 8 — write SUMMARY.md aggregating all diagnostic phases."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def safe_load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return None


def section(title: str, lines: list[str]) -> str:
    return f"## {title}\n\n" + "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: diag_summarize.py <DIAG_ROOT>")
    root = Path(sys.argv[1])

    parts: list[str] = []
    parts.append(f"# Overnight diagnostics — {root.name}\n")
    parts.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n")

    # Status
    status_path = root / "_status.txt"
    if status_path.exists():
        parts.append(section("Phase status (rc per phase)", [
            "```", status_path.read_text().rstrip(), "```"
        ]))

    # Phase 1
    p1 = safe_load(root / "p1_audits" / "prompt_mode.json")
    if p1:
        s = p1.get("summary", {})
        parts.append(section("Phase 1 — Prompt-mode audit (H8)", [
            f"- {s.get('differs_train_vs_val', 'unknown')}",
            f"- {s.get('recommendation', '')}",
        ]))

    # Phase 2
    p2 = safe_load(root / "p2_grpo_group_degeneracy.json")
    if p2:
        h = p2.get("headline", {})
        parts.append(section("Phase 2 — GRPO group-advantage degeneracy (H2)", [
            f"- Steps analyzed: {h.get('n_steps_analyzed')}",
            f"- Mean fraction of degenerate groups (all-same reward): **{(h.get('frac_degenerate_mean') or 0)*100:.1f}%**",
            f"- Max fraction degenerate across steps: {(h.get('frac_degenerate_max') or 0)*100:.1f}%",
            f"- Fraction all-1.0 (Goodhart success): {(h.get('frac_all_one_mean') or 0)*100:.1f}%",
            f"- Mean chosen-rejected margin across steps: {h.get('margin_mean_across_steps')}",
            f"- **Verdict:** {h.get('verdict')}",
        ]))

    # Phase 3
    p3 = safe_load(root / "p3_judge_test_retest.json")
    if p3:
        h = p3.get("headline", {})
        parts.append(section("Phase 3 — Judge test-retest (H1)", [
            f"- N triples: {h.get('n_triples')}, K resamples: {h.get('k_resamples')}",
            f"- Judge per-call σ (test-retest, on margin): **{h.get('test_retest_margin_sigma_mean'):.2f}pt**" if h.get("test_retest_margin_sigma_mean") else "- σ not computed",
            f"- Between-triple margin σ (between-rubric signal): {h.get('between_triple_margin_sigma'):.2f}pt" if h.get("between_triple_margin_sigma") else "",
            f"- Between-triple mean margin: {h.get('between_triple_margin_mean'):.2f}pt" if h.get("between_triple_margin_mean") is not None else "",
            f"- SNR (mean margin / per-call σ): {h.get('snr_margin_mean_over_test_retest_sigma')}" if h.get("snr_margin_mean_over_test_retest_sigma") is not None else "",
            f"- **Verdict:** {h.get('verdict')}",
        ]))

    # Phase 4
    parts_4 = []
    for fname in sorted((root / "p4_ensembling").glob("*_ensembled.json")) if (root / "p4_ensembling").exists() else []:
        d = safe_load(fname)
        if d:
            h = d.get("headline", {})
            parts_4.append(f"### {fname.stem}")
            parts_4.append(f"- K={h.get('K')}  n_rows={h.get('n_rows_kept')}")
            parts_4.append(f"- Individual-run accuracies: {h.get('individual_run_accuracy')}")
            parts_4.append(f"- Avg individual: **{h.get('individual_avg_accuracy')}**")
            parts_4.append(f"- Ensemble accuracy: **{h.get('ensemble_accuracy')}**")
            parts_4.append(f"- Ensembling lift: **{h.get('ensembling_lift_pp')} pp**")
            parts_4.append(f"- Verdict: {h.get('verdict')}")
            parts_4.append("")
    if parts_4:
        parts.append(section("Phase 4 — Judge ensembling re-eval (H1 fix)", parts_4))

    # Phase 5
    parts_5 = []
    p5dir = root / "p5_gpt5_baselines"
    if p5dir.exists():
        for fname in sorted(p5dir.glob("*_gpt5.json")):
            d = safe_load(fname)
            if d:
                rows = d.get("rows", [])
                rewards = [r.get("reward") for r in rows if r.get("reward") is not None]
                acc = sum(rewards) / len(rewards) if rewards else None
                parts_5.append(f"- **{fname.stem}**: GPT-5 accuracy = {acc:.4f} on n={len(rewards)}" if acc else f"- {fname.stem}: no rewards")
    if parts_5:
        parts_5.append("")
        parts_5.append("Compare to existing trained-model GPT-5 results in `logs/.../crossjudge_*_gpt5*.json` to see whether GPT-5 reveals a trained-vs-baseline gap (H3 falsifier).")
        parts.append(section("Phase 5 — GPT-5 cross-judge on UNTRAINED baselines (H3)", parts_5))

    # Phase 6
    p6_path = root / "p6_position_shuffle" / "litbench_trained_flipped.json"
    p6 = safe_load(p6_path) if p6_path.exists() else None
    if p6:
        h = p6.get("headline", {})
        parts.append(section("Phase 6 — Position-shuffle eval (H7)", [
            f"- Flipped accuracy: **{h.get('flipped_accuracy')}** on n={h.get('n_rows')}",
            f"- Verdict: {h.get('verdict')}",
        ]))

    # Phase 7
    p7_path = root / "p7_minimal_prompt" / "litbench_minimal.json"
    p7 = safe_load(p7_path) if p7_path.exists() else None
    if p7:
        h = p7.get("headline", {})
        parts.append(section("Phase 7 — Minimal-prompt baseline (H11)", [
            f"- Minimal-prompt accuracy: **{h.get('minimal_prompt_accuracy')}** (n_parsed={h.get('n_parsed')}/{h.get('n_total')})",
            f"- Verdict: {h.get('verdict')}",
        ]))

    # Cross-cutting interpretation
    parts.append(section("Cross-cutting interpretation", [
        "Pull the verdicts above and decide which scenario applies:",
        "",
        "- **Scenario A (judge-noise + low rollouts)**: H1 confirmed (Phase 3 SNR<1) + H2 confirmed (Phase 2 frac_degen>40%) + Phase 4 ensembling lift ≥1pp. → Run conditional retrain #12 (n=8 rollouts, judge ensembled k=2-4, std-norm GRPO, LR=1e-5, rubric max=1024).",
        "- **Scenario B (judge ceiling + label noise)**: Phase 5 GPT-5 baselines do NOT exceed ~0.72 → judge ceiling is binding. Don't retrain; reframe paper around cross-judge shift + interpretability.",
        "- **Scenario C (eval bug)**: Phase 1 reveals train/val mode mismatch, OR Phase 6 reveals position bias. → Bug-fix and re-run.",
        "",
        "Update `05_04_26_research_plan.md` and `observations_v2.md` with the verdict.",
    ]))

    out_path = root / "SUMMARY.md"
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
