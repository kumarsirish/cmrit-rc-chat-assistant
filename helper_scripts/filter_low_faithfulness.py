"""Filters tests/evaluation-question-answers.json down to just the
questions that scored faithfulness < 1 (or NaN, e.g. a timed-out judge
call) in the last `make eval` run, so that run's fixes can be
re-checked with `make test-ragas-failures` without re-running the
whole (slow) suite.

Requires tests/test-results/judge_results.json (config.evaluation.results_path) to
already exist — run `make eval` first.

Usage:
    python helper_scripts/filter_low_faithfulness.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag._09_common.config import load_config

# Where the filtered question set is written; RAG_EVAL_TEST_DATA_PATH
# points test-ragas at this file (see `make test-ragas-failures`).
FILTERED_TEST_DATA_PATH = "tests/evaluation-question-answers.low-faithfulness.json"


def _needs_recheck(faithfulness) -> bool:
    """True for a score below 1, or NaN — test_judge.py writes NaN when
    a judge call times out/errors rather than raising, so a NaN score
    is just as unverified as a low one and belongs in the recheck set."""
    if faithfulness is None:
        return True
    if isinstance(faithfulness, float) and math.isnan(faithfulness):
        return True
    return faithfulness < 1.0


def select_low_faithfulness_questions(rows: list) -> list:
    """Returns the question text of every row in rows (as written to
    judge_results.json) whose faithfulness needs a recheck."""
    return [row["question"] for row in rows if _needs_recheck(row.get("faithfulness"))]


def filter_test_cases(test_cases: list, questions: list) -> list:
    """Returns the test_cases entries whose "question" is in
    questions, in test_cases's original order."""
    wanted = set(questions)
    return [case for case in test_cases if case["question"] in wanted]


def main():
    config = load_config()
    results_path = Path(config.evaluation.results_path)
    test_data_path = Path(config.evaluation.test_data_path)

    if not results_path.exists():
        print(
            f"No results file at {results_path} — run `make eval` first, "
            "then re-run this."
        )
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as results_file:
        results = json.load(results_file)

    output_path = Path(__file__).resolve().parent.parent / FILTERED_TEST_DATA_PATH

    questions = select_low_faithfulness_questions(results["rows"])
    if not questions:
        # Remove any filtered file left over from a previous run, so
        # `make test-ragas-failures` (which treats output_path's
        # existence as "there's something to re-run") doesn't re-run
        # pytest against a stale question set.
        output_path.unlink(missing_ok=True)
        print(f"Every question in {results_path} scored faithfulness == 1 — nothing to re-check.")
        sys.exit(0)

    with open(test_data_path, "r", encoding="utf-8") as test_data_file:
        test_cases = json.load(test_data_file, strict=False)

    filtered = filter_test_cases(test_cases, questions)
    missing = set(questions) - {case["question"] for case in filtered}
    if missing:
        print(
            f"Warning: {len(missing)} low-faithfulness question(s) from {results_path} "
            f"are no longer in {test_data_path}, skipping them:"
        )
        for question in sorted(missing):
            print(f"  - {question}")

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(filtered, output_file, indent=4)

    print(f"{len(filtered)} question(s) need a recheck, written to {output_path}:")
    for case in filtered:
        print(f"  - {case['question']}")


if __name__ == "__main__":
    main()
