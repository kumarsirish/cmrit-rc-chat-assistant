"""Embedding-based topic detection.

Replaces keyword-overlap matching (the previous find_matching_topic):
comparing literal word sets is brittle against paraphrase — a query
like "who do I email if I'm sick" shares zero words with the topic
name "Leave Information, Application and Approval Process" and would
never match under keyword overlap, even though it is clearly about
that topic. Each topic's name (docs-metadata.json) is embedded once,
at construction time, with the same sentence-transformers model
retrieval already uses for dense search; matching a query then costs
one embedding call plus a handful of cosine-similarity comparisons
against those precomputed vectors, capturing semantic closeness
instead of literal overlap.
"""

import math

# Cosine similarity a query must clear against a topic's name for that
# topic to count as matched. Below this, the query is treated as
# unmatched (see docs_metadata.DocTopic / retriever.py's GLOBAL_MODE
# fallback) rather than forced onto the closest-but-unrelated topic.
DEFAULT_SIMILARITY_THRESHOLD = 0.5


def _cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class EmbeddingTopicMatcher:
    """Finds the DocTopic whose name is most semantically similar to a
    query, via embedding cosine similarity.

    Topic name embeddings are computed once, up front, in a single
    batched embed_texts() call — matching a query never re-embeds the
    topic names.
    """

    def __init__(self, topics: list, embedding_generator, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self._topics = topics
        self._embedding_generator = embedding_generator
        self._similarity_threshold = similarity_threshold
        self._topic_embeddings = (
            embedding_generator.embed_texts([topic.topic for topic in topics]) if topics else []
        )

    def match(self, query: str, query_embedding: list = None):
        """Returns the best-matching DocTopic for query, or None if no
        topic's name clears similarity_threshold (or there are no
        topics at all).

        query_embedding, if given, is used instead of re-embedding
        query — callers (e.g. Retriever) that already computed it for
        dense search can pass it through instead of paying for a
        second embedding call on the same text.
        """
        if not self._topics:
            return None
        if query_embedding is None:
            query_embedding = self._embedding_generator.embed_query(query)

        best_topic = None
        best_score = self._similarity_threshold
        for topic, topic_embedding in zip(self._topics, self._topic_embeddings):
            score = _cosine_similarity(query_embedding, topic_embedding)
            if score > best_score:
                best_score = score
                best_topic = topic
        return best_topic
