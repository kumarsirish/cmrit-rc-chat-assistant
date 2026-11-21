"""Functional (golden-file) retrieval tests, apart from the aggregate
IR metrics in test_retrieval_eval.py.

test_retrieval_eval.py scores the retriever's *average* quality across
the whole test set and only fails if the mean drops below a
threshold, so a regression on one specific question can be averaged
away by the rest. This file instead pins, per question, which
(source_file, page_number) chunks the retriever must surface — see
tests/retrieval-functional-cases.json — and fails that specific case
when it regresses.

Case shape:
    {
        "question": "...",
        "category_override": "..." (optional, matches the UI's category picker),
        "tags": {"category": "...", "difficulty": "..."} (optional, free-form),
        "must_have": [{"source_file": "...", "page_number": 1}, ...],
        "any_of": [[{...}, {...}], ...],
        "nice_to_have": [{...}, ...],
        "no_result_expected": false
    }

- "must_have": every item must appear somewhere in the retrieved
  top_k (unordered — reranker tie-order jitter is not a regression).
  An item may add "text_contains": "15,000" to additionally require
  that substring in the matching chunk's text, for cases where the
  right page has more than one number on it and only one chunk is
  actually correct.
- "any_of": a list of alternative-groups. At least one item from each
  group must appear. Use this where source documents genuinely
  disagree (e.g. RC Norms.docx vs RCPIE FAQ.docx giving different day
  splits for the same review step) instead of asserting both must
  show up.
- "nice_to_have": not asserted — included only in the printed
  recall/precision/NDCG numbers, so a question can carry extra
  plausible-but-not-required sources without turning into a hard
  failure if the retriever doesn't surface them.
- "no_result_expected": true asserts retrieve() comes back empty —
  for out-of-scope questions a correct system should refuse to answer
  (e.g. "what is the cafeteria timing").

Needs the real, ingested Chroma store, so it's excluded from the
default `pytest`/`make test` run (see pytest.ini) and run explicitly:

    make test-retrieval-functional
"""

import json
from pathlib import Path

import pytest

from rag._07_evaluation.baseline import (
    compare_rows_to_baseline,
    compare_to_baseline,
    format_comparison,
    load_baseline_mean_scores,
    load_baseline_rows_by_question,
)
from rag._07_evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k
from rag._09_common.config import load_config
from rag._01_ingestion.embeddings import EmbeddingGenerator
from rag._02_indexing.vector_store import VectorStore
from rag._03_routing.document_metadata import active_doc_topics, load_documents
from rag._04_retrieval.dense_retriever import DenseRetriever
from rag._04_retrieval.reranker import Reranker
from rag._04_retrieval.retrieval_service import RetrievalService

try:
    from rag._04_retrieval.bm25_retriever import BM25Index
except ImportError:  # pragma: no cover - rank_bm25 is a listed dependency
    BM25Index = None

pytestmark = pytest.mark.retrieval_eval

_CASES_PATH = Path(__file__).resolve().parent.parent / "retrieval-functional-cases.json"
_RESULTS_PATH = Path(__file__).resolve().parent.parent / "test-results" / "retrieval_functional_results.json"
_BASELINE_PATH = Path(__file__).resolve().parent.parent / "test-results" / "retrieval_functional_baseline.json"

# Collects one row per case as tests run, for the tag-grouped summary
# printed by test_zz_summary_by_tag below. Module-level because pytest
# gives each parametrized case its own test invocation with no shared
# state otherwise.
_SCORE_ROWS = []


def _load_data():
    with open(_CASES_PATH, "r", encoding="utf-8") as cases_file:
        data = json.load(cases_file)
    return data["cases"] if isinstance(data, dict) else data


def _key(entry: dict) -> str:
    return f"{entry['source_file']}||{entry['page_number']}"


def _matches(chunk, entry: dict) -> bool:
    if chunk.source_file != entry["source_file"] or chunk.page_number != entry["page_number"]:
        return False
    text_contains = entry.get("text_contains")
    return text_contains.lower() in chunk.text.lower() if text_contains else True


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def retrieval_service(config):
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    embedding_generator = EmbeddingGenerator(
        config.embedding.model_name, batch_size=config.embedding.batch_size
    )
    bm25_index = (
        BM25Index.from_vector_store(vector_store)
        if config.retrieval.hybrid.enabled and BM25Index
        else None
    )
    # doc_topics must be passed here the same way dashboard.py passes it
    # to the production RetrievalService — otherwise this test exercises
    # GLOBAL-mode-only behavior and never the PRIMARY path real users'
    # topic-matched queries actually take.
    doc_topics = active_doc_topics(load_documents(config.documents.documents_file))
    return RetrievalService(
        dense_retriever=DenseRetriever(vector_store),
        embedding_generator=embedding_generator,
        top_k=config.retrieval.top_k,
        min_reranker_score=config.retrieval.min_reranker_score,
        min_relevance_score=config.retrieval.min_relevance_score,
        bm25_index=bm25_index,
        hybrid_candidates=config.retrieval.hybrid.candidates,
        dense_weight=config.retrieval.hybrid.dense_weight,
        lexical_weight=config.retrieval.hybrid.lexical_weight,
        reranker=Reranker(config.retrieval.rerank.model) if config.retrieval.rerank.enabled else None,
        rerank_candidates=config.retrieval.rerank.candidates,
        doc_topics=doc_topics,
        primary_top_k=config.retrieval.primary_top_k,
        primary_strong_match_threshold=config.retrieval.primary_strong_match_threshold,
        primary_weak_match_top_k=config.retrieval.primary_weak_match_top_k,
        topic_match_threshold=config.retrieval.topic_match_threshold,
        topic_match_margin=config.retrieval.topic_match_margin,
        topic_match_max_matches=config.retrieval.topic_match_max_matches,
        low_confidence_min_score=config.retrieval.low_confidence_min_score,
    )


@pytest.mark.parametrize("case", _load_data(), ids=lambda case: case["question"][:60])
def test_retrieval_matches_expected(case, retrieval_service, config):
    chunks, _ = retrieval_service.retrieve(case["question"], category_override=case.get("category_override"))

    if case.get("no_result_expected", False):
        assert not chunks, f"Expected no result for an out-of-scope question, got {[(c.source_file, c.page_number) for c in chunks]}"
        return

    must_have = case.get("must_have", [])
    missing = [entry for entry in must_have if not any(_matches(chunk, entry) for chunk in chunks)]
    assert not missing, f"must_have items missing from top_k: {missing}"

    any_of = case.get("any_of", [])
    unsatisfied_groups = [
        group for group in any_of if not any(_matches(chunk, entry) for chunk in chunks for entry in group)
    ]
    assert not unsatisfied_groups, f"None of these alternatives were retrieved: {unsatisfied_groups}"

    # Scoring, for the printed summary below — not asserted. Folds every
    # acceptable source (must_have + every any_of alternative +
    # nice_to_have) into one relevant set so recall/precision/NDCG give
    # credit for whichever alternative actually got retrieved.
    nice_to_have = case.get("nice_to_have", [])
    relevant_keys = {_key(entry) for entry in must_have}
    relevant_keys |= {_key(entry) for group in any_of for entry in group}
    relevant_keys |= {_key(entry) for entry in nice_to_have}
    retrieved_keys = [f"{chunk.source_file}||{chunk.page_number}" for chunk in chunks]
    k = config.retrieval.top_k

    rank_of_first_hit = next(
        (rank for rank, key in enumerate(retrieved_keys, start=1) if key in relevant_keys), None
    )
    _SCORE_ROWS.append(
        {
            "question": case["question"],
            "tags": case.get("tags", {}),
            "recall": recall_at_k(retrieved_keys, relevant_keys, k),
            "precision": precision_at_k(retrieved_keys, relevant_keys, k),
            "reciprocal_rank": reciprocal_rank(retrieved_keys, relevant_keys),
            "ndcg": ndcg_at_k(retrieved_keys, relevant_keys, k),
            "first_hit_rank": rank_of_first_hit,
        }
    )
    print(
        f"\n{case['question'][:70]!r}: recall={_SCORE_ROWS[-1]['recall']:.2f} "
        f"precision={_SCORE_ROWS[-1]['precision']:.2f} mrr={_SCORE_ROWS[-1]['reciprocal_rank']:.2f} "
        f"ndcg={_SCORE_ROWS[-1]['ndcg']:.2f} first_hit_rank={rank_of_first_hit}"
    )


def test_zz_summary_by_tag():
    """Informational only (no assertion) — prints mean recall/MRR/NDCG
    grouped by each tag value collected while the parametrized cases
    above ran, so a regression that's concentrated in one category or
    difficulty level is visible instead of buried in one big average.
    Named to sort/collect after the parametrized test in this same
    file (pytest runs tests in file-definition order, not
    alphabetically, so this only works because it's defined below
    test_retrieval_matches_expected)."""
    if not _SCORE_ROWS:
        pytest.skip("No scored cases collected (parametrized test above didn't run).")

    metric_names = ("recall", "precision", "reciprocal_rank", "ndcg")

    def _mean_by_metric(rows: list) -> dict:
        return {metric: sum(row[metric] for row in rows) / len(rows) for metric in metric_names}

    by_tag_value = {}
    for row in _SCORE_ROWS:
        for tag_name, tag_value in row["tags"].items():
            by_tag_value.setdefault(f"{tag_name}={tag_value}", []).append(row)

    mean_scores = _mean_by_metric(_SCORE_ROWS)
    mean_by_tag = {tag: _mean_by_metric(rows) for tag, rows in by_tag_value.items()}

    print(f"\nOverall ({len(_SCORE_ROWS)} cases):")
    for metric, mean in mean_scores.items():
        print(f"  {metric}: {mean:.3f}")

    for tag, rows in sorted(by_tag_value.items()):
        print(f"\n{tag} ({len(rows)} cases):")
        for metric, mean in mean_by_tag[tag].items():
            print(f"  {metric}: {mean:.3f}")

    baseline_mean_scores = load_baseline_mean_scores(_BASELINE_PATH)
    comparison = compare_to_baseline(mean_scores, baseline_mean_scores)
    if comparison is None:
        print(
            f"\nNo baseline at {_BASELINE_PATH} — run "
            "`make save-retrieval-functional-baseline` to compare future runs against this one."
        )
    else:
        print(f"\nVs baseline ({_BASELINE_PATH}):\n{format_comparison(comparison)}")

    baseline_rows_by_question = load_baseline_rows_by_question(_BASELINE_PATH)
    row_comparison = compare_rows_to_baseline(_SCORE_ROWS, baseline_rows_by_question)
    if row_comparison is not None:
        regressed = [
            entry
            for entry in row_comparison
            if any(m["delta"] < -1e-9 for m in entry["metrics"].values())
        ]
        if regressed:
            print(f"\nPer-question regressions vs baseline ({len(regressed)} of {len(row_comparison)}):")
            for entry in regressed:
                deltas = ", ".join(
                    f"{name}={m['delta']:+.2f}" for name, m in entry["metrics"].items() if m["delta"] < -1e-9
                )
                print(f"  {entry['question'][:70]!r}: {deltas}")
        else:
            print(f"\nNo per-question regressions vs baseline ({len(row_comparison)} questions compared).")

    weak_rows = sorted(
        (row for row in _SCORE_ROWS if row["recall"] < 1.0 or row["reciprocal_rank"] < 0.5),
        key=lambda row: (row["recall"], row["reciprocal_rank"]),
    )
    if weak_rows:
        print(f"\nWeak this run ({len(weak_rows)} of {len(_SCORE_ROWS)} — recall<1.0 or mrr<0.5):")
        for row in weak_rows:
            print(
                f"  {row['question'][:70]!r}: recall={row['recall']:.2f} "
                f"precision={row['precision']:.2f} mrr={row['reciprocal_rank']:.2f} "
                f"first_hit_rank={row['first_hit_rank']}"
            )

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(
            {
                "mean_scores": mean_scores,
                "mean_by_tag": mean_by_tag,
                "rows": _SCORE_ROWS,
                "baseline_comparison": comparison,
                "row_comparison": row_comparison,
            },
            results_file,
            indent=2,
        )
