"""Snapshots the current evaluation results as the baseline future runs
are compared against.

Usage:
    python scripts/save_baseline.py ragas
    python scripts/save_baseline.py retrieval

Copies tests/ragas_results.json -> tests/ragas_baseline.json (or the
retrieval equivalent), both read from config.yaml so this stays in
sync with wherever evaluation.results_path/baseline_path actually
point. Run this right after a `make test-ragas` / `make
test-retrieval-eval` run you want to treat as the reference point —
every later run of that suite will report its delta against this
snapshot (see rag/evaluation/baseline.py).
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.common.config import load_config


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("ragas", "retrieval"):
        print("Usage: python scripts/save_baseline.py [ragas|retrieval]")
        sys.exit(1)

    config = load_config()
    if sys.argv[1] == "ragas":
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
