"""Step 4b: cross-encoder reranking of the retriever's ANN candidates.

Embedding similarity (used by Retriever's initial semantic search)
scores the query and each chunk independently, then compares the two
vectors — fast enough to run over the whole collection, but a weaker
relevance signal. A cross-encoder instead scores the (query, chunk)
pair jointly, which is much more accurate but too slow to run over
more than a handful of candidates. So Retriever over-fetches
(config.retrieval.rerank.candidates) and this module picks the best
top_k of those.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    """Scores (query, chunk) pairs with a cross-encoder model."""

    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def score(self, query: str, chunks: list) -> list:
        """Returns the raw cross-encoder score for each chunk, in the
        same order as chunks (not sorted, not truncated) — used by
        rerank() below, and by Retriever's debug trace logging to show
        the scores before truncation/reordering."""
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        return list(self._model.predict(pairs))

    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        """Returns the top_k chunks, reordered by cross-encoder score
        (most relevant first). chunks is left untouched if empty.
        """
        if not chunks:
            return chunks

        scores = self.score(query, chunks)
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
