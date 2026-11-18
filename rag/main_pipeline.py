"""Wires stages 03-06 (routing, retrieval, generation, response)
together into a runnable RetrievalService/RagPipeline from an
AppConfig.

Shared by rag/_10_ui/dashboard.py (the live Streamlit app),
helper_scripts/log_pipeline_answers.py, and
helper_scripts/run_experiment.py so there is exactly one place that
assembles this wiring instead of several drifting copies.
"""

from rag._01_ingestion.document_loader import list_categories
from rag._01_ingestion.embeddings import EmbeddingGenerator
from rag._01_ingestion.glossary_extractor import load_glossary
from rag._02_indexing.vector_store import VectorStore
from rag._03_routing.category_matcher import EmbeddingCategoryMatcher
from rag._03_routing.docs_metadata import load_doc_topics
from rag._04_retrieval import faq_cache
from rag._04_retrieval.bm25_retriever import BM25Index
from rag._04_retrieval.dense_retriever import DenseRetriever
from rag._04_retrieval.reranker import Reranker
from rag._04_retrieval.retrieval_service import RetrievalService
from rag._05_generation.llm_client import LlmClient
from rag._05_generation.rag_pipeline import RagPipeline


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
    doc_topics = load_doc_topics(config.documents.metadata_file)
    categories = list_categories(config.documents.source_directory)
    category_matcher = EmbeddingCategoryMatcher(
        categories, embedding_generator, similarity_threshold=config.retrieval.category_match_threshold
    )
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
        category_matcher=category_matcher,
    )
    retrieval_service = faq_cache.maybe_wrap(retrieval_service, config.redis)
    return retrieval_service, vector_store


def build_rag_pipeline(config) -> RagPipeline:
    retrieval_service, _ = build_retrieval_service(config)
    llm_client = LlmClient.from_config(config.llm)
    glossary = load_glossary(config.documents.glossary_file)
    return RagPipeline(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
        glossary=glossary,
        no_match_message=config.app.no_match_message,
    )
