"""Runs the real RAG pipeline (retrieval + LLM generation, no RAGAS
judging) for every question in config.evaluation.test_data_path and
writes a plain question/expected/actual log.

Useful for eyeballing what the assistant actually says today without
paying for a RAGAS judge pass:

    python scripts/log_pipeline_answers.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.common.config import load_config
from rag.generation.llm_client import LlmClient
from rag.generation.rag_pipeline import RagPipeline
from rag.indexing.vector_store import VectorStore
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.bm25_retriever import BM25Index
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.reranker import Reranker
from rag.retrieval.retrieval_service import RetrievalService
from rag.routing.docs_metadata import load_doc_topics

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "pipeline_answers.json"


def _load_test_cases(path):
    with open(path, "r", encoding="utf-8") as test_file:
        return json.load(test_file, strict=False)


def build_pipeline(config):
    embedding_generator = EmbeddingGenerator(
        config.embedding.model_name, batch_size=config.embedding.batch_size
    )
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    doc_topics = load_doc_topics(config.documents.metadata_file)
    retrieval_service = RetrievalService(
        dense_retriever=DenseRetriever(vector_store),
        embedding_generator=embedding_generator,
        top_k=config.retrieval.top_k,
        min_reranker_score=config.retrieval.min_reranker_score,
        min_relevance_score=config.retrieval.min_relevance_score,
        bm25_index=BM25Index.from_vector_store(vector_store) if config.retrieval.hybrid.enabled else None,
        hybrid_candidates=config.retrieval.hybrid.candidates,
        dense_weight=config.retrieval.hybrid.dense_weight,
        lexical_weight=config.retrieval.hybrid.lexical_weight,
        reranker=Reranker(config.retrieval.rerank.model) if config.retrieval.rerank.enabled else None,
        rerank_candidates=config.retrieval.rerank.candidates,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=config.retrieval.mapped_document_primary_top_k,
        topic_match_threshold=config.retrieval.topic_match_threshold,
        low_confidence_min_score=config.retrieval.low_confidence_min_score,
    )
    llm_client = LlmClient.from_config(config.llm)
    return RagPipeline(retrieval_service=retrieval_service, llm_client=llm_client)


def main() -> None:
    config = load_config()
    test_cases = _load_test_cases(config.evaluation.test_data_path)
    pipeline = build_pipeline(config)

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
