"""Builds the Chroma vector index from the documents in knowledge_documents.

Run this once before starting the Streamlit app, and again any time
the documents in knowledge_documents change:

    python helper_scripts/ingest.py

Split into independently-testable stages (load, chunk, build metadata,
embed, store) rather than one function, so e.g. metadata-building can
be unit-tested without touching the embedding model or Chroma. The
BM25 index is deliberately not built/persisted here — it's rebuilt in
memory from the vector store at application startup (see
rag/_04_retrieval/bm25_retriever.py), so there's nothing for ingestion to write for it.

config/glossary.json is hand-maintained, not regenerated here — see
rag/_01_ingestion/glossary_extractor.py if it ever needs re-extracting
from the source PDFs.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import PurePosixPath

from rag._09_common.config import AppConfig, load_config
from rag._03_routing.document_metadata import DocTopic, active_filename_topic_map, load_documents
from rag._02_indexing.vector_store import VectorStore
from rag._01_ingestion.chunker import Chunk, chunk_documents
from rag._01_ingestion.document_loader import find_document_files, load_all_documents
from rag._01_ingestion.embeddings import EmbeddingGenerator

# category used for a document sitting directly in source_directory
# with no category subfolder — matches admin_page.py's convention for
# describing such a file in its documents listing.
UNCATEGORIZED = "(uncategorized)"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("rag").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def load_pages(config: AppConfig) -> list:
    """Stage 1+2 (load + validate): every usable PageDocument from
    config.documents.source_directory. See document_loader.py for what
    "usable" means and how a bad PDF is handled."""
    logger.info("Loading documents from %s ...", config.documents.source_directory)
    pages = load_all_documents(config.documents.source_directory, config.documents.archived_subfolder_name)
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


def dedupe_chunks(chunks: list) -> list:
    """EXPERIMENTAL (added 2026-09-25 to fix boilerplate crowding out
    real content in dense search — see git log for the investigation;
    revert this function and its call in run_ingestion() to roll back).

    Drops any chunk whose text (whitespace-normalized) is identical to
    an earlier chunk's. Several policy documents (the Faculty, HoD,
    and Student handbooks) share the same front-matter verbatim (e.g.
    "About CMRIT", trust history, management bios) — indexing it once
    per document means that boilerplate's near-identical embedding
    scores well against ANY query mentioning the institution by name,
    outranking document-specific content that only appears once (see
    e.g. the Staff Welfare Measures fee-concession chunk, previously
    buried at rank 252/1141 in dense search behind repeated "About
    CMRIT" paragraphs). Keeps the first occurrence, including its
    source_file/page for citation.
    """
    seen_texts = set()
    deduped = []
    dropped = 0
    for chunk in chunks:
        normalized_text = " ".join(chunk.text.split())
        if normalized_text in seen_texts:
            dropped += 1
            continue
        seen_texts.add(normalized_text)
        deduped.append(chunk)
    if dropped:
        logger.info("Dropped %d duplicate chunk(s) also found in another document.", dropped)
    return deduped


def build_chunk_metadata(config: AppConfig, chunks: list) -> tuple:
    """Stage 4 (build metadata): the single place chunk metadata is
    assembled for storage — ids, and one metadata dict per chunk
    carrying source file, page range, section, category/location, and
    (when the source file matches an active document record)
    topic/canonical link.

    Returns (ids, metadatas), aligned index-for-index with chunks.
    """
    logger.info("Loading topic metadata from %s ...", config.documents.documents_file)
    records = load_documents(config.documents.documents_file)
    filename_topic_map = active_filename_topic_map(records)
    location_map = _build_filename_location_map(
        config.documents.source_directory, config.documents.archived_subfolder_name
    )

    ids = [_chunk_id(chunk) for chunk in chunks]
    metadatas = [_chunk_metadata(chunk, filename_topic_map, location_map) for chunk in chunks]
    return ids, metadatas


def _build_filename_location_map(source_directory: str, archived_subfolder_name: str) -> dict:
    """Maps basename -> (category, source_path) for every currently-
    scanned document, by re-deriving it from each file's position
    under source_directory (category = its immediate parent folder's
    name, or UNCATEGORIZED for a file with no category subfolder;
    source_path = its path relative to source_directory).

    Returns an empty map if source_directory doesn't exist yet (same
    tolerance find_document_files itself has no use for here, since an
    ingestion run with no documents already fails later in load_pages
    — this just needs to not crash before that point).

    A basename shared by files in two different categories collides
    here (whichever is scanned last wins) — not a new limitation this
    introduces: filename is already the sole cross-document identity
    key everywhere else in this codebase (chunk IDs below, the
    documents file's filename->topic map, PRIMARY search's
    source_file filter), just newly visible once category is tracked.
    """
    root = Path(source_directory)
    if not root.exists():
        return {}

    location_map = {}
    for path in find_document_files(source_directory, archived_subfolder_name):
        category = path.parent.relative_to(root).parts[0] if path.parent != root else UNCATEGORIZED
        source_path = str(PurePosixPath(path.relative_to(root)))
        existing = location_map.get(path.name)
        if existing and existing[0] != category:
            logger.warning(
                "%r exists in both category %r and %r — chunk metadata will use %r "
                "(whichever was scanned last).",
                path.name, existing[0], category, category,
            )
        location_map[path.name] = (category, source_path)
    return location_map


def _chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source_file}::page-{chunk.page_number}::chunk-{chunk.chunk_index}"


def _chunk_metadata(chunk: Chunk, filename_topic_map: dict, location_map: dict) -> dict:
    matched_topic: DocTopic = filename_topic_map.get(chunk.source_file.strip().casefold())
    category, source_path = location_map.get(chunk.source_file, (UNCATEGORIZED, chunk.source_file))
    document_type = Path(chunk.source_file).suffix.lstrip(".").lower()
    return {
        "source_file": chunk.source_file,
        "page_number": chunk.page_number,
        "page_end": chunk.page_end,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "topic": matched_topic.topic if matched_topic else "",
        # matched_topic.link can genuinely be None — e.g. a document
        # uploaded via the Admin tab (admin_page.py's
        # _register_single_upload) doesn't collect a link, so
        # document_metadata.register_upload never writes a link at
        # all. Chroma rejects None outright as a metadata value (same
        # reasoning as the "keywords" guard below), so this must be ""
        # not None whenever the topic has no link, same as an
        # unmatched chunk already gets.
        "source_link": (matched_topic.link or "") if matched_topic else "",
        # Curated terms (job titles, synonyms, ...) a chunk's own prose
        # doesn't contain but queries about it might use — e.g. "Roles
        # and Responsibilities.docx.pdf" never says "Director". Stored
        # on every chunk from this topic's document so search_text()
        # can fold them into what gets embedded/BM25-indexed, without
        # ever touching the chunk text shown to the LLM or in
        # citations. Chroma metadata values can't be None.
        "keywords": (matched_topic.keywords or "") if matched_topic else "",
        # Category-aware retrieval (rag._04_retrieval.retrieval_service)
        # filters on this.
        "category": category,
        # Same value as source_file, under the name an external
        # consumer of this metadata would expect.
        "source": chunk.source_file,
        "source_path": source_path,
        "page": chunk.page_number,
        # Same value already used as this chunk's Chroma record id (see
        # _chunk_id) — stored again as metadata so it's retrievable
        # alongside the chunk's text/other metadata in one lookup.
        "chunk_id": _chunk_id(chunk),
        # Optional document classification; defaults to the file
        # extension since nothing finer-grained is curated yet.
        "document_type": document_type,
    }


def search_text(chunk_text: str, keywords: str) -> str:
    """Text actually embedded/BM25-indexed for a chunk: its own text
    plus any curated keywords for its document's topic (see
    _chunk_metadata). Retrieval matches on this; everything shown to
    the LLM or cited to the user still uses the chunk's plain text."""
    if not keywords:
        return chunk_text
    return f"{chunk_text}\n\nKeywords: {keywords}"


def embed_chunks(config: AppConfig, chunks: list, metadatas: list, progress_callback=None) -> list:
    """Stage 5 (embed): one vector per chunk, embedded from search_text
    (chunk text + any curated keywords for its topic), not the chunk's
    raw text alone — see search_text().

    progress_callback(message, fraction), if given, is reported after
    every embedding batch (via EmbeddingGenerator.embed_texts's own
    progress_callback), not just once at the end of this whole stage —
    this is normally the single longest-running stage of ingestion, so
    only reporting progress at its start/end would leave a caller's
    progress bar frozen for most of a rebuild.
    """
    logger.info("Loading embedding model %s ...", config.embedding.model_name)
    embedding_generator = EmbeddingGenerator(
        config.embedding.model_name, batch_size=config.embedding.batch_size
    )

    texts = [
        search_text(chunk.text, metadata["keywords"]) for chunk, metadata in zip(chunks, metadatas)
    ]
    logger.info("Embedding %d chunks with batch size %d ...", len(chunks), config.embedding.batch_size)

    def _on_batch(done, total):
        logger.info("Embedded %d/%d chunks.", done, total)
        if progress_callback:
            progress_callback(f"Embedding chunks ({done}/{total})", done / total)

    try:
        return embedding_generator.embed_texts(
            texts, progress_callback=_on_batch if progress_callback else None
        )
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


def load_ingestion_stats(config: AppConfig) -> dict:
    """Returns the last successful run's {"last_duration_seconds",
    "last_run_at"}, or {} if save_ingestion_stats has never run (e.g.
    a brand-new deployment with no prior rebuild) — used by the Admin
    tab to show an estimated time before starting a new rebuild."""
    stats_path = Path(config.documents.ingestion_stats_file)
    if not stats_path.exists():
        return {}
    try:
        with open(stats_path, "r", encoding="utf-8") as stats_file:
            return json.load(stats_file)
    except (OSError, json.JSONDecodeError):
        return {}


def save_ingestion_stats(config: AppConfig, duration_seconds: float) -> None:
    """Persists this run's duration as the estimate for the next one
    (see load_ingestion_stats)."""
    stats_path = Path(config.documents.ingestion_stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(
            {
                "last_duration_seconds": duration_seconds,
                "last_run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            stats_file,
            indent=2,
        )
        stats_file.write("\n")


def run_ingestion(config: AppConfig = None, progress_callback=None) -> None:
    """config, if given, is used as-is instead of a fresh load_config()
    call — lets a caller that already has a (possibly overridden)
    config, e.g. helper_scripts/run_experiment.py building a dedicated
    index for a non-default embedding model, reuse this exact pipeline
    instead of duplicating it.

    progress_callback(message: str, fraction: float), if given, is
    called at each stage boundary (and, for the embedding stage, after
    every batch — see embed_chunks) with fraction in [0, 1] — used by
    the Admin tab's "Rebuild index" button to drive a real progress bar
    and a live log of what's currently happening, instead of a bare
    spinner. The stage boundaries below are a rough, fixed split, not
    measured per document set — embedding is given the bulk of the
    range (0.15-0.90) since it's normally by far the longest stage.
    """
    if config is None:
        config = load_config()

    def _report(message, fraction):
        if progress_callback:
            progress_callback(message, fraction)

    start = time.perf_counter()

    _report("Loading documents...", 0.0)
    pages: list = load_pages(config)
    _report(f"Loaded {len(pages)} page(s).", 0.05)

    chunks: list = chunk_pages(config, pages)
    _report(f"Created {len(chunks)} chunk(s).", 0.08)

    chunks = dedupe_chunks(chunks)  # EXPERIMENTAL — see dedupe_chunks docstring
    _report("Deduplicated chunks.", 0.10)

    ids, metadatas = build_chunk_metadata(config, chunks)
    _report("Built chunk metadata.", 0.15)

    embeddings = embed_chunks(config, chunks, metadatas, progress_callback=progress_callback)
    _report(f"Embedded {len(chunks)} chunk(s).", 0.90)

    vector_store = store_chunks(config, ids, chunks, embeddings, metadatas)
    _report("Wrote the vector index.", 0.97)

    total_seconds = time.perf_counter() - start
    save_ingestion_stats(config, total_seconds)

    logger.info("Done. Collection now has %d chunks.", vector_store.count())
    _report(f"Done — collection now has {vector_store.count()} chunks.", 1.0)


if __name__ == "__main__":
    run_ingestion()
