"""Baseline comparison for evaluation runs.

A "baseline" is just a snapshot of a previous evaluation results file
(tests/test-results/judge_results.json or tests/test-results/retrieval_results.json), saved to
a separate path via `make save-eval-baseline` (see
helper_scripts/save_baseline.py). Every subsequent `make eval` run loads
that snapshot, if present, and reports each metric's delta against it
so regressions/improvements are visible without manually diffing two
JSON files.
"""

import json
from pathlib import Path


def load_baseline_mean_scores(baseline_path: str) -> dict | None:
    """Returns the {metric: score} mapping from a saved baseline file,
    or None if no baseline has been saved yet."""
    path = Path(baseline_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as baseline_file:
        return json.load(baseline_file)["mean_scores"]


def compare_to_baseline(mean_scores: dict, baseline_mean_scores: dict | None) -> dict | None:
    """Returns {metric: {"current", "baseline", "delta"}} for metrics
    present in both, or None if there is no baseline to compare against."""
    if baseline_mean_scores is None:
        return None
    return {
        name: {
            "current": current,
            "baseline": baseline_mean_scores[name],
            "delta": current - baseline_mean_scores[name],
        }
        for name, current in mean_scores.items()
        if name in baseline_mean_scores
    }


def format_comparison(comparison: dict) -> str:
    """Renders a comparison dict (from compare_to_baseline) as
    human-readable lines for printing in test output."""
    lines = []
    for name, values in comparison.items():
        delta = values["delta"]
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
        lines.append(
            f"  {name}: {values['current']:.3f} vs baseline {values['baseline']:.3f} "
            f"({arrow} {delta:+.3f})"
        )
    return "\n".join(lines)
