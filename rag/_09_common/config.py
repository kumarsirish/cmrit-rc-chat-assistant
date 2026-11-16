"""Loads application configuration from config.yaml.

Every other module reads its settings through this module instead of
hard-coding values, so behaviour (which LLM, which embedding model,
timeouts, etc.) can be changed in one place.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"


@dataclass
class OllamaConfig:
    base_url: str
    model: str
    temperature: float
    request_timeout_seconds: int
    num_thread: int
    num_predict: int


@dataclass
class GroqConfig:
    model: str
    temperature: float
    request_timeout_seconds: int
    max_tokens: int
    api_key_env: str


@dataclass
class LlmConfig:
    provider: str
    ollama: OllamaConfig
    groq: GroqConfig


@dataclass
class EmbeddingConfig:
    model_name: str
    batch_size: int


@dataclass
class ChromaConfig:
    persist_directory: str
    collection_name: str


@dataclass
class DocumentsConfig:
    source_directory: str
    metadata_file: str
    # Acronym -> full-phrase glossary, hand-maintained (see
    # rag._01_ingestion.glossary_extractor for the one-off extraction
    # tool it was seeded from). Used to expand jargon-free questions
    # before retrieval — see rag._03_routing.query_expander.
    glossary_file: str


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    separators: list


@dataclass
class RerankConfig:
    enabled: bool
    model: str
    candidates: int


@dataclass
class HybridConfig:
    enabled: bool
    candidates: int
    dense_weight: float
    lexical_weight: float


@dataclass
class RetrievalConfig:
    top_k: int
    min_reranker_score: float
    min_relevance_score: float
    mapped_document_primary_top_k: int
    topic_match_threshold: float
    hybrid: HybridConfig
    rerank: RerankConfig
    # Floor for the low-confidence fallback in
    # retrieval_service._filter_relevant_by_reranker: when nothing
    # clears min_reranker_score, the single best-scoring candidate is
    # still returned (flagged low_confidence=True) as long as its
    # score is at least this — below it, the query is treated as
    # genuinely unanswerable and no chunk is returned at all.
    low_confidence_min_score: float = 0.05


@dataclass
class SessionConfig:
    timeout_minutes: int
    warning_seconds_before_timeout: int
    max_history_messages: int


@dataclass
class RetrievalEvalConfig:
    # k is intentionally not duplicated here — retrieval eval scores
    # Recall/Precision/NDCG @ config.retrieval.top_k, the same k the
    # app actually uses at answer time.
    min_recall: float
    min_mrr: float
    results_path: str
    baseline_path: str


@dataclass
class EvaluationConfig:
    judge_provider: str
    judge_model: str
    judge_temperature: float
    judge_timeout_seconds: int
    judge_max_workers: int
    test_data_path: str
    metrics: list
    min_score: float
    results_path: str
    baseline_path: str
    retrieval: RetrievalEvalConfig


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class RedisConfig:
    # Global cache of frequently-repeated questions' retrieved chunks
    # (see rag/_04_retrieval/faq_cache.py) — off by default since it's
    # optional infra (nothing else here requires a Redis server).
    enabled: bool
    host: str
    port: int
    db: int
    password: str
    # A question must be asked (exact match, after normalization) at
    # least this many times before its chunks get cached.
    faq_cache_min_hits: int
    # Safety-net expiry for a cached entry. reset_if_enabled() (called
    # after every ingestion run) is what normally invalidates the
    # cache; this just bounds how stale an entry could get if that
    # ever doesn't happen.
    cache_ttl_seconds: int


@dataclass
class AppConfig:
    llm: LlmConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    documents: DocumentsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    session: SessionConfig
    evaluation: EvaluationConfig
    database: DatabaseConfig
    redis: RedisConfig


def _resolve_path(value: str) -> str:
    """Turns a relative path from config.yaml into an absolute path."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Reads config.yaml and returns a typed AppConfig object.

    The MySQL password can be overridden with the RAG_DB_PASSWORD
    environment variable so it does not have to live in the file.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    llm_raw = dict(raw["llm"])
    llm_raw["ollama"] = OllamaConfig(**llm_raw["ollama"])
    llm_raw["groq"] = GroqConfig(**llm_raw["groq"])
    llm_cfg = LlmConfig(**llm_raw)
    embedding_cfg = EmbeddingConfig(**raw["embedding"])

    chroma_raw = raw["chroma"]
    chroma_cfg = ChromaConfig(
        persist_directory=_resolve_path(chroma_raw["persist_directory"]),
        collection_name=chroma_raw["collection_name"],
    )

    documents_raw = raw["documents"]
    documents_cfg = DocumentsConfig(
        source_directory=_resolve_path(documents_raw["source_directory"]),
        metadata_file=_resolve_path(documents_raw["metadata_file"]),
        glossary_file=_resolve_path(documents_raw["glossary_file"]),
    )

    chunking_cfg = ChunkingConfig(**raw["chunking"])

    retrieval_raw = dict(raw["retrieval"])
    retrieval_raw["hybrid"] = HybridConfig(**retrieval_raw["hybrid"])
    retrieval_raw["rerank"] = RerankConfig(**retrieval_raw["rerank"])
    retrieval_cfg = RetrievalConfig(**retrieval_raw)

    session_cfg = SessionConfig(**raw["session"])

    evaluation_raw = dict(raw["evaluation"])
    # Both overridable so a debugging run (see helper_scripts/filter_low_faithfulness.py
    # / `make test-eval-failures`) can point the judge eval at a filtered
    # question set and a scratch results file, without touching
    # config.yaml or clobbering the full-suite tests/test-results/judge_results.json.
    evaluation_raw["test_data_path"] = _resolve_path(
        os.environ.get("RAG_EVAL_TEST_DATA_PATH", evaluation_raw["test_data_path"])
    )
    evaluation_raw["results_path"] = _resolve_path(
        os.environ.get("RAG_EVAL_RESULTS_PATH", evaluation_raw["results_path"])
    )
    evaluation_raw["baseline_path"] = _resolve_path(evaluation_raw["baseline_path"])
    retrieval_eval_raw = dict(evaluation_raw["retrieval"])
    retrieval_eval_raw["results_path"] = _resolve_path(retrieval_eval_raw["results_path"])
    retrieval_eval_raw["baseline_path"] = _resolve_path(retrieval_eval_raw["baseline_path"])
    evaluation_raw["retrieval"] = RetrievalEvalConfig(**retrieval_eval_raw)
    evaluation_cfg = EvaluationConfig(**evaluation_raw)

    database_raw = dict(raw["database"])
    database_raw["password"] = os.environ.get(
        "RAG_DB_PASSWORD", database_raw["password"]
    )
    database_cfg = DatabaseConfig(**database_raw)

    redis_cfg = RedisConfig(**raw["redis"])

    return AppConfig(
        llm=llm_cfg,
        embedding=embedding_cfg,
        chroma=chroma_cfg,
        documents=documents_cfg,
        chunking=chunking_cfg,
        retrieval=retrieval_cfg,
        session=session_cfg,
        evaluation=evaluation_cfg,
        database=database_cfg,
        redis=redis_cfg,
    )
