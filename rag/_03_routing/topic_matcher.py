"""Embedding-based topic detection.

Matches a query to every sufficiently-similar document topic (see
document_metadata.DocTopic) by semantic similarity, not literal word
overlap: a query like "who do I email if I'm sick" shares zero words
with the topic name "Leave Information, Application and Approval
Process" but is clearly about it. A query can match more than one
topic — e.g. two different forms filed under the same category — so
this returns every topic that both clears the absolute threshold AND
is competitive with the single best match (see similarity_margin
below), not just whichever topics happen to clear the threshold on
their own. Each topic's name is embedded once, at construction time,
with the same sentence-transformers model retrieval already uses for
dense search; matching a query then costs one embedding call plus a
handful of cosine-similarity comparisons against those precomputed
vectors.

The matched topics drive two things downstream (see
retrieval_service.py): which document(s), if any, retrieval should
search first (docs_filename), and which canonical link(s) to show the
user, even for a topic with no locally-indexed PDF.
"""

import math

# Cosine similarity a query must clear against a topic's name for that
# topic to count as matched. Below this, the query is treated as
# unmatched (see document_metadata.DocTopic / retriever.py's GLOBAL_MODE
# fallback) rather than forced onto the closest-but-unrelated topic.
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# How far below the single best-scoring topic's score another topic
# can fall and still count as matched alongside it. Without this, a
# query where ANY topic happens to clear the absolute threshold admits
# every one of them regardless of how much worse it scored than the
# best — observed in practice matching 9-18 "relevant" topics for a
# single-document question, which then let PRIMARY search dilute
# itself across mostly-wrong documents and crowd out the one correct
# answer. Kept topics must be within this margin of the best score, so
# only genuinely competitive matches (e.g. two forms in the same
# category that are both plausibly what the query means) survive
# alongside it, not every topic that merely cleared the floor.
DEFAULT_SIMILARITY_MARGIN = 0.10

# Hard cap on how many topics match_all() ever returns, independent of
# the margin above — a second line of defense against PRIMARY fanning
# out across more documents than a single question could reasonably
# be about.
DEFAULT_MAX_MATCHES = 3


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
    """Finds every DocTopic similar enough to a query to count as
    matched — a query can legitimately be about more than one document
    (e.g. two forms in the same category), so this returns every topic
    clearing the threshold rather than just the single best one.

    Topic name embeddings are computed once, up front, in a single
    batched embed_texts() call, so matching a query only needs to
    embed the query itself.
    """

    def __init__(
        self,
        topics: list,
        embedding_generator,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        similarity_margin: float = DEFAULT_SIMILARITY_MARGIN,
        max_matches: int = DEFAULT_MAX_MATCHES,
    ):
        self._topics = topics
        self._embedding_generator = embedding_generator
        self._similarity_threshold = similarity_threshold
        self._similarity_margin = similarity_margin
        self._max_matches = max_matches

        # Each topic can contribute multiple candidate phrases (name +
        # keywords); flatten them into one batched embed_texts() call,
        # remembering which topic each phrase embedding belongs to so
        # match_all() can take a per-topic max over its phrases.
        self._phrase_topic_indices = []
        all_phrases = []
        for topic_index, topic in enumerate(topics):
            for phrase in _phrases_for(topic):
                all_phrases.append(phrase)
                self._phrase_topic_indices.append(topic_index)
        self._phrase_embeddings = embedding_generator.embed_texts(all_phrases) if all_phrases else []

    def match_all(self, query: str, query_embedding: list = None) -> list:
        """Returns every DocTopic competitive with the best match for
        query, most similar first — empty if nothing clears
        similarity_threshold (or there are no topics at all).

        A topic survives two filters, not just the threshold: its
        score must clear similarity_threshold, AND it must be within
        similarity_margin of the single best-scoring topic found —
        plus a hard cap of max_matches regardless of how many are
        within the margin. This is what keeps a query that's really
        about one document from also admitting a handful of other
        topics that merely cleared the absolute threshold on their
        own without being genuinely competitive with the best match —
        see DEFAULT_SIMILARITY_MARGIN's docstring for the real
        over-matching case this guards against.

        query_embedding, if given, is used instead of re-embedding
        query — callers (e.g. RetrievalService) that already computed
        it for dense search can pass it through instead of paying for
        a second embedding call on the same text.
        """
        if not self._topics:
            return []
        if query_embedding is None:
            query_embedding = self._embedding_generator.embed_query(query)

        best_score_by_topic = {}
        for topic_index, phrase_embedding in zip(self._phrase_topic_indices, self._phrase_embeddings):
            score = _cosine_similarity(query_embedding, phrase_embedding)
            if score >= self._similarity_threshold:
                best_score_by_topic[topic_index] = max(score, best_score_by_topic.get(topic_index, score))

        if not best_score_by_topic:
            return []

        margin_floor = max(best_score_by_topic.values()) - self._similarity_margin
        qualifying = {
            topic_index: score for topic_index, score in best_score_by_topic.items() if score >= margin_floor
        }

        ranked_indices = sorted(qualifying, key=qualifying.get, reverse=True)[: self._max_matches]
        return [self._topics[topic_index] for topic_index in ranked_indices]
