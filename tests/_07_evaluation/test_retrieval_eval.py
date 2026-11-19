"""Classic IR evaluation of the retriever alone, apart from the
LLM-judged generation metrics in test_judge.py.

For every case in tests/evaluation-question-answers.json that has a
"relevant_sources" list (the ground-truth source files for that
question), runs the real hybrid Retriever against the real Chroma
store and scores Recall@K/Precision@K/MRR/NDCG@K (see
rag/_07_evaluation/retrieval_metrics.py) against those documents.

No LLM call is involved (Retriever.retrieve() is dense+BM25+rerank,
not generation), so this is much faster than the judge half of
`make eval` and needs only Ollama-free local models — but it still
needs the real, ingested Chroma store, so it's excluded from the
default `pytest`/`make test` run (see pytest.ini) and run explicitly:

    make eval
"""

import json

import pytest

from rag._09_common.config import load_config
from rag._07_evaluation.baseline import compare_to_baseline, format_comparison, load_baseline_mean_scores
from rag._07_evaluation.evaluate_retrieval import evaluate_retrieval
from rag._02_indexing.vector_store import VectorStore
from rag._01_ingestion.embeddings import EmbeddingGenerator
from rag._04_retrieval.dense_retriever import DenseRetriever
from rag._04_retrieval.reranker import Reranker
from rag._04_retrieval.retrieval_service import RetrievalService
from rag._03_routing.document_metadata import active_doc_topics, load_documents

try:
    from rag._04_retrieval.bm25_retriever import BM25Index
except ImportError:  # pragma: no cover - rank_bm25 is a listed dependency
    BM25Index = None

pytestmark = pytest.mark.retrieval_eval


def _load_test_cases(path):
    with open(path, "r", encoding="utf-8") as test_file:
        return json.load(test_file, strict=False)


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
    # to the production RetrievalService — otherwise this evaluation
    # silently scores GLOBAL-mode-only behavior and never exercises the
    # PRIMARY path real users' topic-matched queries actually take.
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
        topic_match_threshold=config.retrieval.topic_match_threshold,
        low_confidence_min_score=config.retrieval.low_confidence_min_score,
    )


def test_retrieval_quality_meets_minimum_thresholds(config, retrieval_service):
    test_cases = [
        case
        for case in _load_test_cases(config.evaluation.test_data_path)
        if "relevant_sources" in case
    ]
    assert test_cases, "No test cases with 'relevant_sources' to evaluate retrieval against."

    result = evaluate_retrieval(retrieval_service, test_cases, config.retrieval.top_k)
    mean_scores = result["mean_scores"]

    print(f"\nRetrieval mean scores (k={result['k']}): {mean_scores}")
    print(f"Retrieval mean scores by category: {result['mean_by_category']}")

    baseline_mean_scores = load_baseline_mean_scores(config.evaluation.retrieval.baseline_path)
    comparison = compare_to_baseline(mean_scores, baseline_mean_scores)
    if comparison is None:
        print(
            f"\nNo baseline at {config.evaluation.retrieval.baseline_path} — run "
            "`make save-eval-baseline` to compare future runs against this one."
        )
    else:
        print(
            f"\nVs baseline ({config.evaluation.retrieval.baseline_path}):\n"
            f"{format_comparison(comparison)}"
        )
    result["baseline_comparison"] = comparison

    with open(config.evaluation.retrieval.results_path, "w", encoding="utf-8") as results_file:
        json.dump(result, results_file, indent=2)

    failing = {
        "recall": mean_scores["recall"] < config.evaluation.retrieval.min_recall,
        "reciprocal_rank": mean_scores["reciprocal_rank"] < config.evaluation.retrieval.min_mrr,
    }
    failing = {name: mean_scores[name] for name, failed in failing.items() if failed}
    assert not failing, (
        f"Retrieval metrics below configured minimums: {failing} "
        f"(all mean scores: {mean_scores})"
    )
