"""Step 3b: Vector storage.

Thin wrapper around a local, on-disk Chroma collection, built on
LangChain's Chroma vector store. Ingestion uses add_chunks() to store
pre-embedded policy chunks (embeddings are computed once, in batch, by
EmbeddingGenerator during ingestion); retrieval uses
similarity_search_with_relevance_scores() to fetch the chunks closest
to an already-embedded user question.
"""

import logging
import re

from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when the underlying Chroma collection can't be opened or
    queried."""


def _to_chroma_where(filter: dict):
    """Converts a plain {key: value} exact-match filter into the shape
    Chroma's `where` argument accepts: None when empty, the single
    equality clause directly when there's one key, and an explicit
    "$and" of equality clauses when there are several (Chroma rejects a
    bare multi-key dict)."""
    if not filter:
        return None
    if len(filter) == 1:
        key, value = next(iter(filter.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in filter.items()]}


class VectorStore:
    """Persistent Chroma collection holding embedded policy chunks."""

    def __init__(self, persist_directory: str, collection_name: str):
        try:
            self._store = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                collection_configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception as exc:
            logger.error("Could not open Chroma collection '%s' at %s: %s", collection_name, persist_directory, exc)
            raise VectorStoreError(f"Could not open Chroma collection '{collection_name}': {exc}") from exc

    def reset(self) -> None:
        """Deletes all existing entries, used before a fresh ingestion run."""
        existing_ids = self._store._collection.get()["ids"]
        if existing_ids:
            self._store._collection.delete(ids=existing_ids)

    def add_chunks(self, ids: list, texts: list, embeddings: list, metadatas: list) -> None:
        """Stores chunk text, embeddings, and metadata in the collection."""
        self._store._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self._store._collection.count()

    def count_with_filter(self, filter: dict) -> int:
        """Returns how many stored chunks match filter (same exact-match
        shape as similarity_search_with_relevance_scores's filter)."""
        result = self._store._collection.get(where=_to_chroma_where(filter), include=[])
        return len(result["ids"])

    def similarity_search_with_relevance_scores(
        self, query_embedding: list, top_k: int, filter: dict = None
    ) -> list:
        """Returns the top_k (Document, relevance_score) pairs closest to
        query_embedding, most relevant first. relevance_score is on a
        [0, 1] scale (1 = identical).

        filter, if given, is an exact-match metadata filter (e.g.
        {"source_file": "leave.pdf"}) — Chroma's `where` clause when one
        key is given, `{"$and": [...]}` of equality clauses when several
        are, since Chroma doesn't accept a plain multi-key dict.

        Despite its name, Chroma's similarity_search_by_vector_with_
        relevance_scores returns raw cosine DISTANCE, not relevance —
        its own docstring says "lower score represents more
        similarity". Passing that straight through as "relevance"
        inverted every threshold comparison in the app: a near-perfect
        match (small distance, e.g. 0.30) read as low relevance and
        got dropped, while a mediocre match (larger distance, e.g.
        0.50) read as more relevant and was kept. Converting here
        (distance -> 1 - distance) is what makes "higher score = more
        relevant" actually true for callers.
        """
        try:
            results = self._store.similarity_search_by_vector_with_relevance_scores(
                query_embedding, k=top_k, filter=_to_chroma_where(filter)
            )
        except Exception as exc:
            logger.error("Chroma similarity search failed: %s", exc)
            raise VectorStoreError(f"Chroma similarity search failed: {exc}") from exc
        return [(document, 1.0 - distance) for document, distance in results]

    def get_all_documents(self) -> list:
        """Returns every chunk currently stored as (text, metadata)
        pairs. Used to build the in-memory BM25 lexical index (see
        bm25_index.py) once at startup from whatever ingestion last
        wrote to this collection.
        """
        result = self._store._collection.get(include=["documents", "metadatas"])
        return list(zip(result["documents"], result["metadatas"]))

    def find_chunks_containing_any(self, keywords: list, limit: int) -> dict:
        """Returns chunks whose text literally contains any of the given
        keywords (case-insensitive substring match), with their
        embeddings so callers can still rank/score the matches.

        Used as a fallback when semantic search alone comes up empty
        (e.g. a short query built around a proper noun/brand name the
        embedding model has little signal for) but the term is still
        genuinely present in the documents.
        """
        if not keywords:
            return {"documents": [], "metadatas": [], "embeddings": []}

        # Chroma's $contains is case-sensitive (it sets
        # `PRAGMA case_sensitive_like = ON` in its SQLite backend), so
        # a lowercased keyword would never match a capitalized word
        # like "President" in the source text. $regex is honored by
        # Python's re module underneath, so "(?i)" gives a real
        # case-insensitive match instead.
        regex_clauses = [{"$regex": f"(?i){re.escape(keyword)}"} for keyword in keywords]
        where_document = (
            regex_clauses[0] if len(regex_clauses) == 1 else {"$or": regex_clauses}
        )
        return self._store._collection.get(
            where_document=where_document,
            limit=limit,
            include=["documents", "metadatas", "embeddings"],
        )
