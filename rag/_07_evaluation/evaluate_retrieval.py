"""Runs the classic IR evaluation (Recall/Precision/MRR/NDCG @ k) for
a RetrievalService against a set of labeled test cases.

Extracted as a plain function so both the pytest pass/fail gate
(tests/_07_evaluation/test_retrieval_eval.py) and the MLflow experiment
script (helper_scripts/run_experiment.py) share exactly one scoring
implementation, instead of two copies that can drift apart.
"""

import statistics

from rag._07_evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k

_METRIC_NAMES = ("recall", "precision", "reciprocal_rank", "ndcg")


def _mean_by_metric(rows: list) -> dict:
    return {metric: statistics.mean(row[metric] for row in rows) for metric in _METRIC_NAMES}


def evaluate_retrieval(retrieval_service, test_cases: list, k: int) -> dict:
    """Scores retrieval_service against test_cases (each a dict with
    "question", "relevant_sources", and optionally "category" —
    callers should only pass cases that actually have
    "relevant_sources").

    Returns {"k", "mean_scores", "mean_by_category", "rows"}: the same
    shape previously written straight to tests/test-results/retrieval_results.json,
    now also what helper_scripts/run_experiment.py logs to MLflow.
    """
    rows = []
    for case in test_cases:
        chunks, _ = retrieval_service.retrieve(case["question"])
        retrieved_sources = [chunk.source_file for chunk in chunks]
        relevant_sources = set(case["relevant_sources"])
        rows.append(
            {
                "category": case.get("category", "uncategorized"),
                "question": case["question"],
                "recall": recall_at_k(retrieved_sources, relevant_sources, k),
                "precision": precision_at_k(retrieved_sources, relevant_sources, k),
                "reciprocal_rank": reciprocal_rank(retrieved_sources, relevant_sources),
                "ndcg": ndcg_at_k(retrieved_sources, relevant_sources, k),
            }
        )

    by_category = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    return {
        "k": k,
        "mean_scores": _mean_by_metric(rows),
        "mean_by_category": {category: _mean_by_metric(category_rows) for category, category_rows in by_category.items()},
        "rows": rows,
    }
