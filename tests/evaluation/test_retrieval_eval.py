"""Classic IR evaluation of the retriever alone, apart from the
LLM-judged generation metrics in test_ragas_judge.py.

For every case in tests/evaluation-question-answers.json that has a
"relevant_sources" list (the ground-truth source files for that
question), runs the real hybrid Retriever against the real Chroma
store and scores Recall@K/Precision@K/MRR/NDCG@K (see
rag/evaluation/retrieval_metrics.py) against those documents.

No LLM call is involved (Retriever.retrieve() is dense+BM25+rerank,
not generation), so this is much faster than `make test-ragas` and
needs only Ollama-free local models — but it still needs the real,
ingested Chroma store, so it's excluded from the default `pytest`/
`make test` run (see pytest.ini) and run explicitly:

    make test-retrieval-eval
"""

import json
import statistics

import pytest

from rag.common.config import load_config
from rag.evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k
from rag.indexing.vector_store import VectorStore
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.reranker import Reranker
from rag.retrieval.retrieval_service import RetrievalService
from rag.routing.docs_metadata import load_doc_topics

try:
    from rag.retrieval.bm25_retriever import BM25Index
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
    # MAPPED_DOCUMENT path real users' topic-matched queries actually take.
    doc_topics = load_doc_topics(config.documents.metadata_file)
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
        mapped_document_primary_top_k=config.retrieval.mapped_document_primary_top_k,
        topic_match_threshold=config.retrieval.topic_match_threshold,
    )


def test_retrieval_quality_meets_minimum_thresholds(config, retrieval_service):
    test_cases = [
        case
        for case in _load_test_cases(config.evaluation.test_data_path)
        if "relevant_sources" in case
    ]
    assert test_cases, "No test cases with 'relevant_sources' to evaluate retrieval against."

    k = config.retrieval.top_k
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

    mean_scores = {
        metric: statistics.mean(row[metric] for row in rows)
        for metric in ("recall", "precision", "reciprocal_rank", "ndcg")
    }
    by_category = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    mean_by_category = {
        category: {
            metric: statistics.mean(row[metric] for row in category_rows)
            for metric in ("recall", "precision", "reciprocal_rank", "ndcg")
        }
        for category, category_rows in by_category.items()
    }

    print(f"\nRetrieval mean scores (k={k}): {mean_scores}")
    print(f"Retrieval mean scores by category: {mean_by_category}")

    with open(config.evaluation.retrieval.results_path, "w", encoding="utf-8") as results_file:
        json.dump(
            {"k": k, "mean_scores": mean_scores, "mean_by_category": mean_by_category, "rows": rows},
            results_file,
            indent=2,
        )

    failing = {
        "recall": mean_scores["recall"] < config.evaluation.retrieval.min_recall,
        "reciprocal_rank": mean_scores["reciprocal_rank"] < config.evaluation.retrieval.min_mrr,
    }
    failing = {name: mean_scores[name] for name, failed in failing.items() if failed}
    assert not failing, (
        f"Retrieval metrics below configured minimums: {failing} "
        f"(all mean scores: {mean_scores})"
    )
