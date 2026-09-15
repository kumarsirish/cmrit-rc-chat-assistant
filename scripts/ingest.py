"""Builds the Chroma vector index from the PDFs in policies_docs.

Run this once before starting the Streamlit app, and again any time
the documents in policies_docs change:

    python scripts/ingest.py

Split into independently-testable stages (load, chunk, build metadata,
embed, store) rather than one function, so e.g. metadata-building can
be unit-tested without touching the embedding model or Chroma. The
BM25 index is deliberately not built/persisted here — it's rebuilt in
memory from the vector store at application startup (see
rag/retrieval/bm25_retriever.py), so there's nothing for ingestion to write for it.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.common.config import AppConfig, load_config
from rag.routing.docs_metadata import DocTopic, build_filename_topic_map, load_doc_topics
from rag.indexing.vector_store import VectorStore
from rag.ingestion.chunker import Chunk, chunk_documents
from rag.ingestion.document_loader import load_all_documents
from rag.ingestion.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("rag").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def load_pages(config: AppConfig) -> list:
    """Stage 1+2 (load + validate): every usable PageDocument from
    config.documents.source_directory. See document_loader.py for what
    "usable" means and how a bad PDF is handled."""
    logger.info("Loading PDFs from %s ...", config.documents.source_directory)
    pages = load_all_documents(config.documents.source_directory)
    logger.info("Loaded %d pages.", len(pages))
    return pages


def chunk_pages(config: AppConfig, pages: list) -> list:
    """Stage 3 (chunk): every page's text split into Chunk objects."""
    logger.info("Chunking %d pages ...", len(pages))
    chunks = chunk_documents(
        pages,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
    )
    logger.info("Created %d chunks.", len(chunks))
    return chunks


def build_chunk_metadata(config: AppConfig, chunks: list) -> tuple:
    """Stage 4 (build metadata): the single place chunk metadata is
    assembled for storage — ids, and one metadata dict per chunk
    carrying source file, page range, section, and (when the source
    file matches a docs-metadata.json entry) topic/canonical link.

    Returns (ids, metadatas), aligned index-for-index with chunks.
    """
    logger.info("Loading topic metadata from %s ...", config.documents.metadata_file)
    doc_topics = load_doc_topics(config.documents.metadata_file)
    filename_topic_map = build_filename_topic_map(doc_topics)

    ids = [_chunk_id(chunk) for chunk in chunks]
    metadatas = [_chunk_metadata(chunk, filename_topic_map) for chunk in chunks]
    return ids, metadatas


def _chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source_file}::page-{chunk.page_number}::chunk-{chunk.chunk_index}"


def _chunk_metadata(chunk: Chunk, filename_topic_map: dict) -> dict:
    matched_topic: DocTopic = filename_topic_map.get(chunk.source_file.strip().casefold())
    return {
        "source_file": chunk.source_file,
        "page_number": chunk.page_number,
        "page_end": chunk.page_end,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "topic": matched_topic.topic if matched_topic else "",
        "source_link": matched_topic.link if matched_topic else "",
    }


def embed_chunks(config: AppConfig, chunks: list) -> list:
    """Stage 5 (embed): one vector per chunk's text."""
    logger.info("Loading embedding model %s ...", config.embedding.model_name)
    embedding_generator = EmbeddingGenerator(
        config.embedding.model_name, batch_size=config.embedding.batch_size
    )

    logger.info("Embedding %d chunks with batch size %d ...", len(chunks), config.embedding.batch_size)
    try:
        return embedding_generator.embed_texts([chunk.text for chunk in chunks])
    except Exception as exc:
        raise RuntimeError(f"Embedding failed for {len(chunks)} chunks: {exc}") from exc


def store_chunks(config: AppConfig, ids: list, chunks: list, embeddings: list, metadatas: list) -> VectorStore:
    """Stage 6 (store): replaces the Chroma collection's contents with
    the freshly embedded chunks."""
    logger.info("Writing to Chroma collection '%s' ...", config.chroma.collection_name)
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    texts = [chunk.text for chunk in chunks]
    try:
        vector_store.reset()
        vector_store.add_chunks(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
    except Exception as exc:
        raise RuntimeError(f"Writing to Chroma collection failed: {exc}") from exc
    return vector_store


def run_ingestion() -> None:
    config = load_config()

    pages: list = load_pages(config)
    chunks: list = chunk_pages(config, pages)
    ids, metadatas = build_chunk_metadata(config, chunks)
    embeddings = embed_chunks(config, chunks)
    vector_store = store_chunks(config, ids, chunks, embeddings, metadatas)

    logger.info("Done. Collection now has %d chunks.", vector_store.count())


if __name__ == "__main__":
    run_ingestion()
