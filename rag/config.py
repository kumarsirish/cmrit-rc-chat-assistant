"""Loads application configuration from config.yaml.

Every other module reads its settings through this module instead of
hard-coding values, so behaviour (which LLM, which embedding model,
timeouts, etc.) can be changed in one place.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class OllamaConfig:
    base_url: str
    model: str
    temperature: float
    request_timeout_seconds: int
    num_thread: int


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


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    separators: list


@dataclass
class RetrievalConfig:
    top_k: int
    min_relevance_score: float


@dataclass
class SessionConfig:
    timeout_minutes: int
    warning_seconds_before_timeout: int
    max_history_messages: int


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class AppConfig:
    ollama: OllamaConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    documents: DocumentsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    session: SessionConfig
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

    ollama_cfg = OllamaConfig(**raw["ollama"])
    embedding_cfg = EmbeddingConfig(**raw["embedding"])

    chroma_raw = raw["chroma"]
    chroma_cfg = ChromaConfig(
        persist_directory=_resolve_path(chroma_raw["persist_directory"]),
        collection_name=chroma_raw["collection_name"],
    )

    documents_raw = raw["documents"]
    documents_cfg = DocumentsConfig(
        source_directory=_resolve_path(documents_raw["source_directory"])
    )

    chunking_cfg = ChunkingConfig(**raw["chunking"])
    retrieval_cfg = RetrievalConfig(**raw["retrieval"])
    session_cfg = SessionConfig(**raw["session"])

    database_raw = dict(raw["database"])
    database_raw["password"] = os.environ.get(
        "RAG_DB_PASSWORD", database_raw["password"]
    )
    database_cfg = DatabaseConfig(**database_raw)

    return AppConfig(
        ollama=ollama_cfg,
        embedding=embedding_cfg,
        chroma=chroma_cfg,
        documents=documents_cfg,
        chunking=chunking_cfg,
        retrieval=retrieval_cfg,
        session=session_cfg,
        database=database_cfg,
    )
