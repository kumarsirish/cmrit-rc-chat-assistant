"""Embedding-based topic detection.

Matches a query to a topic in docs-metadata.json by semantic
similarity, not literal word overlap: a query like "who do I email if
I'm sick" shares zero words with the topic name "Leave Information,
Application and Approval Process" but is clearly about it. Each
topic's name is embedded once, at construction time, with the same
sentence-transformers model retrieval already uses for dense search;
matching a query then costs one embedding call plus a handful of
cosine-similarity comparisons against those precomputed vectors.

The matched topic drives two things downstream (see
retrieval_service.py and docs_metadata.py): which document, if any,
retrieval should search first (docs_filename), and which canonical
link to show the user, even for topics with no locally-indexed PDF.
"""

import math

# Cosine similarity a query must clear against a topic's name for that
# topic to count as matched. Below this, the query is treated as
# unmatched (see docs_metadata.DocTopic / retriever.py's GLOBAL_MODE
# fallback) rather than forced onto the closest-but-unrelated topic.
DEFAULT_SIMILARITY_THRESHOLD = 0.5


def _phrases_for(topic) -> list:
    """Candidate phrases representing a topic: its name plus any
    curated keywords (job titles, synonyms, ...) that broaden what a
    query can match on beyond the topic name's literal words.

    Kept as separate short phrases rather than one concatenated blob:
    sentence-transformer mean-pooling dilutes a long blob's embedding
    toward its average meaning, which can pull a specific keyword's
    signal (e.g. "Director") below a short, topically-adjacent topic
    name's similarity. Scoring each phrase independently and taking
    the max (see EmbeddingTopicMatcher.match) avoids that dilution.
    """
    phrases = [topic.topic]
    if topic.keywords:
        phrases.extend(keyword.strip() for keyword in topic.keywords.split(",") if keyword.strip())
    return phrases


def _cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class EmbeddingTopicMatcher:
    """Finds the DocTopic (given in docs-metadta.json) similar to a
    query.

    Topic name embeddings are computed once, up front, in a single
    batched embed_texts() call, so matching a query only needs to
    embed the query itself.
    """

    def __init__(self, topics: list, embedding_generator, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self._topics = topics
        self._embedding_generator = embedding_generator
        self._similarity_threshold = similarity_threshold

        # Each topic can contribute multiple candidate phrases (name +
        # keywords); flatten them into one batched embed_texts() call,
        # remembering which topic each phrase embedding belongs to so
        # match() can take a per-topic max over its phrases.
        self._phrase_topic_indices = []
        all_phrases = []
        for topic_index, topic in enumerate(topics):
            for phrase in _phrases_for(topic):
                all_phrases.append(phrase)
                self._phrase_topic_indices.append(topic_index)
        self._phrase_embeddings = embedding_generator.embed_texts(all_phrases) if all_phrases else []

    def match(self, query: str, query_embedding: list = None):
        """Returns the best-matching DocTopic for query, or None if no
        topic's name/keyword phrase clears similarity_threshold (or
        there are no topics at all).

        query_embedding, if given, is used instead of re-embedding
        query — callers (e.g. Retriever) that already computed it for
        dense search can pass it through instead of paying for a
        second embedding call on the same text.
        """
        if not self._topics:
            return None
        if query_embedding is None:
            query_embedding = self._embedding_generator.embed_query(query)

        best_topic_index = None
        best_score = self._similarity_threshold
        for topic_index, phrase_embedding in zip(self._phrase_topic_indices, self._phrase_embeddings):
            score = _cosine_similarity(query_embedding, phrase_embedding)
            if score > best_score:
                best_score = score
                best_topic_index = topic_index
        return self._topics[best_topic_index] if best_topic_index is not None else None
