"""Dense (embedding) search over the Chroma vector store.

Thin wrapper around VectorStore's LangChain-backed similarity search,
so the rest of retrieval doesn't need to know about Chroma directly.
"""


class DenseRetriever:
    """Runs dense search against one Chroma vector store."""

    def __init__(self, vector_store):
        self._vector_store = vector_store

    def search(self, query_embedding: list, top_k: int, filter: dict = None) -> list:
        """Returns (Document, score) pairs, most relevant first.

        query_embedding is computed once by the caller (see
        retrieval_service.py) and reused here, instead of embedding the
        query text again.
        """
        return self._vector_store.similarity_search_with_relevance_scores(
            query_embedding, top_k, filter=filter
        )
