"""Runs the real RAG pipeline (retrieval + LLM generation, no RAGAS
judging) for every question in config.evaluation.test_data_path and
writes a plain question/expected/actual log.

Useful for eyeballing what the assistant actually says today without
paying for a RAGAS judge pass:

    python helper_scripts/log_pipeline_answers.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag._09_common.config import load_config
from rag.pipeline_factory import build_rag_pipeline

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "pipeline_answers.json"


def _load_test_cases(path):
    with open(path, "r", encoding="utf-8") as test_file:
        return json.load(test_file, strict=False)


def main() -> None:
    config = load_config()
    test_cases = _load_test_cases(config.evaluation.test_data_path)
    pipeline = build_rag_pipeline(config)

    rows = []
    total = len(test_cases)
    for index, case in enumerate(test_cases, start=1):
        question = case["question"]
        print(f"[{index}/{total}] {question[:80]}", flush=True)
        start = time.perf_counter()
        result = pipeline.answer(question)
        elapsed = time.perf_counter() - start
        confidence_note = " [LOW-CONFIDENCE]" if result.low_confidence else ""
        print(f"    -> {len(result.answer_text)} chars in {elapsed:.1f}s{confidence_note}", flush=True)
        rows.append(
            {
                "category": case.get("category", "normal"),
                "question": question,
                "expected_answer": case.get("answer", ""),
                "actual_answer": result.answer_text,
                "sources": [source.source_file for source in result.sources] if result.sources else [],
                "low_confidence": result.low_confidence,
                "seconds": round(elapsed, 3),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "test_data_path": config.evaluation.test_data_path,
                "rows": rows,
            },
            output_file,
            indent=2,
        )
    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
