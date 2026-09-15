"""Cross-encoder reranking of the hybrid retriever's candidates.

A cross-encoder scores the (query, chunk) pair jointly, which is more
accurate than comparing two independent embeddings, but too slow to
run over the whole collection — so retrieval_service.py only sends it
a small candidate pool.

Uses LangChain's HuggingFaceCrossEncoder to load and run the model.
Scoring stays a separate step from picking the final chunks: LangChain's
CrossEncoderReranker sorts, truncates, and drops the scores in one
step, but retrieval_service.py needs the scores first, to filter by
min_reranker_score before truncating to a top_k count.
"""

import operator

from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class Reranker:
    """Scores (query, chunk) pairs with a cross-encoder model."""

    def __init__(self, model_name: str):
        self._model = HuggingFaceCrossEncoder(model_name=model_name)

    def score(self, query: str, chunks: list) -> list:
        """Returns one cross-encoder score per chunk, in the same
        order as chunks (not sorted, not truncated)."""
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        return list(self._model.score(pairs))

    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        """Returns the top_k chunks, reordered by cross-encoder score
        (most relevant first). chunks is left untouched if empty.
        """
        if not chunks:
            return chunks

        scores = self.score(query, chunks)
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=operator.itemgetter(0), reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
