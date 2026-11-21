"""Snapshots the current evaluation results as the baseline future runs
are compared against.

Usage:
    python helper_scripts/save_baseline.py deepeval
    python helper_scripts/save_baseline.py retrieval
    python helper_scripts/save_baseline.py retrieval-functional

Copies tests/test-results/judge_results.json -> tests/test-results/judge_baseline.json (or the
retrieval/retrieval-functional equivalent). deepeval/retrieval paths
are read from config.yaml so they stay in sync with wherever
evaluation.results_path/baseline_path actually point;
retrieval-functional's are fixed (see tests/_07_evaluation/
test_retrieval_functional.py, which isn't part of config.yaml's
evaluation block). Called by `make save-eval-baseline` (deepeval +
retrieval) or `make save-retrieval-functional-baseline` right after a
run you want to treat as the reference point — every later run
reports its delta against this snapshot (see
rag/_07_evaluation/baseline.py).
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag._09_common.config import load_config

_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests" / "test-results"


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("deepeval", "retrieval", "retrieval-functional"):
        print("Usage: python helper_scripts/save_baseline.py [deepeval|retrieval|retrieval-functional]")
        sys.exit(1)

    if sys.argv[1] == "retrieval-functional":
        results_path = _TESTS_DIR / "retrieval_functional_results.json"
        baseline_path = _TESTS_DIR / "retrieval_functional_baseline.json"
    else:
        config = load_config()
        if sys.argv[1] == "deepeval":
            results_path = Path(config.evaluation.results_path)
            baseline_path = Path(config.evaluation.baseline_path)
        else:
            results_path = Path(config.evaluation.retrieval.results_path)
            baseline_path = Path(config.evaluation.retrieval.baseline_path)

    if not results_path.exists():
        print(f"No results file at {results_path} — run the evaluation first.")
        sys.exit(1)

    shutil.copyfile(results_path, baseline_path)
    print(f"Saved baseline: {results_path} -> {baseline_path}")


if __name__ == "__main__":
    main()
