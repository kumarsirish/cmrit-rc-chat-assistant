"""Step 4: Retrieval from the Chroma vector database.

Given a user question, embeds it with the same embedding model used
during ingestion and fetches the closest policy chunks. Chunks below
the configured relevance threshold are dropped so the pipeline can
recognise "this question is not covered by our policies" instead of
forcing an answer out of weakly related text.

If semantic search comes back with nothing above the threshold, a
keyword fallback looks for a literal, case-insensitive match of the
query's significant words (see keyword_extraction.py) in the chunk
text. This catches short queries about proper nouns/brand names
(e.g. "what is darwinbox") that the embedding model has too little
signal to place semantically, even though the term is genuinely
present in the documents.
"""

import math
import operator
from dataclasses import dataclass

from rag.embeddings import EmbeddingGenerator
from rag.keyword_extraction import extract_keywords
from rag.vector_store import VectorStore

# How many keyword-match candidates to pull before ranking them by
# cosine similarity and keeping the best top_k.
KEYWORD_FALLBACK_CANDIDATE_MULTIPLIER = 5


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    relevance_score: float
    match_type: str = "semantic"


def _cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


class Retriever:
    """Finds the policy chunks most relevant to a user question."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        top_k: int,
        min_relevance_score: float,
    ):
        self._vector_store = vector_store
        self._embedding_generator = embedding_generator
        self._top_k = top_k
        self._min_relevance_score = min_relevance_score

    def retrieve(self, query: str) -> list:
        """Returns relevant chunks for query, most relevant first."""
        query_embedding = self._embedding_generator.embed_query(query)

        semantic_chunks = self._semantic_search(query_embedding)
        if semantic_chunks:
            return semantic_chunks

        return self._keyword_fallback_search(query, query_embedding)

    def _semantic_search(self, query_embedding: list) -> list:
        results = self._vector_store.query(query_embedding, self._top_k)

        retrieved_chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            relevance_score = 1.0 - distance
            if relevance_score < self._min_relevance_score:
                continue
            retrieved_chunks.append(
                RetrievedChunk(
                    text=text,
                    source_file=metadata["source_file"],
                    page_number=metadata["page_number"],
                    relevance_score=relevance_score,
                    match_type="semantic",
                )
            )
        return retrieved_chunks

    def _keyword_fallback_search(self, query: str, query_embedding: list) -> list:
        keywords = extract_keywords(query)
        if not keywords:
            return []

        candidate_limit = self._top_k * KEYWORD_FALLBACK_CANDIDATE_MULTIPLIER
        results = self._vector_store.find_chunks_containing_any(keywords, candidate_limit)

        scored_chunks = []
        for text, metadata, embedding in zip(
            results["documents"], results["metadatas"], results["embeddings"]
        ):
            relevance_score = _cosine_similarity(query_embedding, embedding)
            scored_chunks.append(
                RetrievedChunk(
                    text=text,
                    source_file=metadata["source_file"],
                    page_number=metadata["page_number"],
                    relevance_score=relevance_score,
                    match_type="keyword",
                )
            )

        scored_chunks.sort(key=operator.attrgetter("relevance_score"), reverse=True)
        return scored_chunks[: self._top_k]
