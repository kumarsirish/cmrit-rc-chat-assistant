"""Step 3b: Vector storage.

Thin wrapper around a local, on-disk ChromaDB collection. Ingestion
uses add_chunks() to store embedded policy chunks; retrieval uses
query() to fetch the chunks closest to a user's question.
"""

import re

import chromadb


class VectorStore:
    """Persistent Chroma collection holding embedded policy chunks."""

    def __init__(self, persist_directory: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Deletes all existing entries, used before a fresh ingestion run."""
        existing_ids = self._collection.get()["ids"]
        if existing_ids:
            self._collection.delete(ids=existing_ids)

    def add_chunks(self, ids: list, texts: list, embeddings: list, metadatas: list) -> None:
        """Stores chunk text, embeddings, and metadata in the collection."""
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self._collection.count()

    def query(self, query_embedding: list, top_k: int) -> dict:
        """Returns the top_k chunks closest to query_embedding.

        Chroma returns cosine distance (0 = identical, higher = less
        similar); callers that want a similarity score should convert
        it with 1 - distance.
        """
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

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
        return self._collection.get(
            where_document=where_document,
            limit=limit,
            include=["documents", "metadatas", "embeddings"],
        )
