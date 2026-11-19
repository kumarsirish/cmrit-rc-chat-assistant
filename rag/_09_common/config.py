"""Loads application configuration from config.yaml.

Every other module reads its settings through this module instead of
hard-coding values, so behaviour (which LLM, which embedding model,
timeouts, etc.) can be changed in one place.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify_app_name(app_name: str) -> str:
    """Turns app.name into a safe identifier — lowercased, every run of
    non-alphanumeric characters collapsed to a single underscore, and
    trimmed. Used to derive the Chroma collection name and the MySQL
    database name from app.name, so multiple deployments sharing one
    Chroma/MySQL instance (different config.yaml, same servers) each
    get their own collection/database instead of mixing data —
    renaming app.name is then the one place that changes which
    collection/database this deployment reads and writes."""
    slug = _SLUG_PATTERN.sub("_", app_name.strip().lower()).strip("_")
    return slug or "app"


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
    # Derived from app.name (see slugify_app_name), not read from
    # config.yaml directly — so each deployment's collection is named
    # after its own app, even when several deployments share one
    # Chroma persist_directory/server.
    collection_name: str


@dataclass
class AppUiConfig:
    # Shown as the page title and intro text in the Streamlit UI, so
    # deploying this app for a different document set doesn't require
    # touching dashboard.py.
    name: str
    description: str
    # Clickable example questions shown in the chat sidebar — deployment-
    # specific, so they live in config rather than a hardcoded list.
    example_queries: list
    # Reply used when retrieval finds nothing above the relevance
    # threshold (see rag._06_response.response_processor). Configurable
    # so it can be worded for whatever document set this deployment
    # indexes, instead of being locked to one domain's phrasing.
    no_match_message: str
    # Every chat session is attributed to this identity since there is
    # no real login — a placeholder, not a security boundary.
    default_user_email: str


@dataclass
class AdminConfig:
    # Shared password gating the Admin tab's upload/supersede/rebuild-
    # index controls. Overridable via RAG_ADMIN_PASSWORD so it doesn't
    # have to live in the committed file, same pattern as
    # DatabaseConfig.password/RAG_DB_PASSWORD.
    password: str


@dataclass
class DocumentsConfig:
    source_directory: str
    # One record per document — category, upload time, active/archived
    # status and supersession history, plus the topic/link fields used
    # to detect which document a query is about. See
    # rag._03_routing.document_metadata.
    documents_file: str
    # Acronym -> full-phrase glossary, hand-maintained (see
    # rag._01_ingestion.glossary_extractor for the one-off extraction
    # tool it was seeded from). Used to expand jargon-free questions
    # before retrieval — see rag._03_routing.query_expander.
    glossary_file: str
    # Subfolder name (under each category folder) that a superseded
    # document is moved into. Ingestion's directory scan skips any path
    # under a folder with this name, so an archived document stops being
    # retrievable without needing to touch the vector store directly.
    archived_subfolder_name: str = "_archived"
    # Records the duration of the last successful run_ingestion() call
    # (see helper_scripts/ingest.py's save_ingestion_stats) — read by
    # the Admin tab's "Rebuild index" button to show an estimated time
    # before the next run, since there's no other way to predict how
    # long a given document set will take to re-embed.
    ingestion_stats_file: str = "./config/ingestion_stats.json"


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
    primary_top_k: int
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
    # Derived from app.name (see slugify_app_name), not read from
    # config.yaml directly — so each deployment gets its own MySQL
    # database (sessions/messages/timing all scoped to it) even when
    # several deployments share one MySQL server, instead of mixing
    # session history and "estimated answer time" stats across apps.
    database: str


@dataclass
class AppConfig:
    app: AppUiConfig
    admin: AdminConfig
    llm: LlmConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    documents: DocumentsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    session: SessionConfig
    evaluation: EvaluationConfig
    database: DatabaseConfig


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

    app_cfg = AppUiConfig(**raw["app"])

    admin_raw = dict(raw["admin"])
    admin_raw["password"] = os.environ.get("RAG_ADMIN_PASSWORD", admin_raw["password"])
    admin_cfg = AdminConfig(**admin_raw)

    llm_raw = dict(raw["llm"])
    llm_raw["ollama"] = OllamaConfig(**llm_raw["ollama"])
    llm_raw["groq"] = GroqConfig(**llm_raw["groq"])
    llm_cfg = LlmConfig(**llm_raw)
    embedding_cfg = EmbeddingConfig(**raw["embedding"])

    chroma_raw = raw["chroma"]
    chroma_cfg = ChromaConfig(
        persist_directory=_resolve_path(chroma_raw["persist_directory"]),
        collection_name=slugify_app_name(app_cfg.name),
    )

    documents_raw = raw["documents"]
    documents_cfg = DocumentsConfig(
        source_directory=_resolve_path(documents_raw["source_directory"]),
        documents_file=_resolve_path(documents_raw["documents_file"]),
        glossary_file=_resolve_path(documents_raw["glossary_file"]),
        archived_subfolder_name=documents_raw.get("archived_subfolder_name", "_archived"),
        ingestion_stats_file=_resolve_path(
            documents_raw.get("ingestion_stats_file", "./config/ingestion_stats.json")
        ),
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
    database_raw["database"] = f"{slugify_app_name(app_cfg.name)}_sessions"
    database_cfg = DatabaseConfig(**database_raw)

    return AppConfig(
        app=app_cfg,
        admin=admin_cfg,
        llm=llm_cfg,
        embedding=embedding_cfg,
        chroma=chroma_cfg,
        documents=documents_cfg,
        chunking=chunking_cfg,
        retrieval=retrieval_cfg,
        session=session_cfg,
        evaluation=evaluation_cfg,
        database=database_cfg,
    )
