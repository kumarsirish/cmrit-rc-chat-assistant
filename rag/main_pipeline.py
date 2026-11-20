"""Wires stages 03-06 (routing, retrieval, generation, response)
together into a runnable RetrievalService/RagPipeline from an
AppConfig.

Shared by rag/_10_ui/dashboard.py (the live Streamlit app),
helper_scripts/log_pipeline_answers.py, and
helper_scripts/run_experiment.py so there is exactly one place that
assembles this wiring instead of several drifting copies.
"""

from rag._01_ingestion.embeddings import EmbeddingGenerator
from rag._01_ingestion.glossary_extractor import load_glossary
from rag._02_indexing.vector_store import VectorStore
from rag._03_routing.document_metadata import active_doc_topics, load_documents
from rag._04_retrieval.bm25_retriever import BM25Index
from rag._04_retrieval.dense_retriever import DenseRetriever
from rag._04_retrieval.reranker import Reranker
from rag._04_retrieval.retrieval_service import RetrievalService
from rag._05_generation.llm_client import LlmClient
from rag._05_generation.rag_pipeline import RagPipeline
from rag._05_generation.similar_question_finder import find_similar_question
from rag._09_common.query_logger import log_query


def build_retrieval_service(config) -> tuple:
    """Returns (retrieval_service, vector_store) — vector_store is
    returned alongside since some callers (e.g. run_experiment.py) need
    it directly for things retrieval_service doesn't expose, like
    vector_store.count()."""
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    embedding_generator = EmbeddingGenerator(
        config.embedding.model_name, batch_size=config.embedding.batch_size
    )
    doc_topics = active_doc_topics(load_documents(config.documents.documents_file))
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
        primary_top_k=config.retrieval.primary_top_k,
        primary_strong_match_threshold=config.retrieval.primary_strong_match_threshold,
        primary_weak_match_top_k=config.retrieval.primary_weak_match_top_k,
        topic_match_threshold=config.retrieval.topic_match_threshold,
        topic_match_margin=config.retrieval.topic_match_margin,
        topic_match_max_matches=config.retrieval.topic_match_max_matches,
        low_confidence_min_score=config.retrieval.low_confidence_min_score,
    )
    return retrieval_service, vector_store


def build_rag_pipeline(config, session_store=None) -> RagPipeline:
    """session_store, if given (the live app passes its own — see
    dashboard.get_rag_pipeline), backs the "did you mean this
    already-answered question?" suggestion (see
    rag._05_generation.similar_question_finder) with real past
    questions from MySQL. Deliberately optional: evaluation/experiment
    callers (helper_scripts/run_experiment.py, log_pipeline_answers.py)
    don't pass one, so building a pipeline for them never requires a
    live MySQL connection — RagPipeline falls back to its no-op default
    for similar_question_finder in that case.
    """
    retrieval_service, _ = build_retrieval_service(config)
    llm_client = LlmClient.from_config(config.llm)
    glossary = load_glossary(config.documents.glossary_file)

    similar_question_finder = None
    if session_store is not None:
        threshold = config.retrieval.similar_question_threshold
        embedding_generator = retrieval_service.embedding_generator

        def similar_question_finder(question: str) -> str:
            candidate_questions = session_store.get_grounded_questions()
            return find_similar_question(question, candidate_questions, embedding_generator, threshold)

    return RagPipeline(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
        glossary=glossary,
        no_match_message=config.app.no_match_message,
        log_query=log_query,
        similar_question_finder=similar_question_finder,
    )
