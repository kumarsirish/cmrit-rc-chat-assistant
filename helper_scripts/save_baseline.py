"""Snapshots the current evaluation results as the baseline future runs
are compared against.

Usage:
    python helper_scripts/save_baseline.py deepeval
    python helper_scripts/save_baseline.py retrieval

Copies tests/test-results/judge_results.json -> tests/test-results/judge_baseline.json (or the
retrieval equivalent), both read from config.yaml so this stays in
sync with wherever evaluation.results_path/baseline_path actually
point. Called by `make save-eval-baseline` right after a `make eval`
run you want to treat as the reference point — every later `make eval`
run will report its delta against this snapshot (see
rag/_07_evaluation/baseline.py).
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag._09_common.config import load_config


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("deepeval", "retrieval"):
        print("Usage: python helper_scripts/save_baseline.py [deepeval|retrieval]")
        sys.exit(1)

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
