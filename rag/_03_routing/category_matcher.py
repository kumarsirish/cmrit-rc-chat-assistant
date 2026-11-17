"""Embedding-based category detection.

Mirrors topic_matcher.EmbeddingTopicMatcher's approach (embed each
candidate name once, compare a query's embedding against those
precomputed vectors by cosine similarity) but for category subfolders
under knowledge_documents instead of docs-metadata.json topics — and,
unlike topic matching, returns every category that clears the
threshold rather than just the single best one, since a question can
legitimately span more than one category (e.g. a question about both
funding and approval forms).

Used by retrieval_service.py as a coarser fallback for whatever would
otherwise be a plain GLOBAL (unscoped) search: zero matches means the
question can't be confidently placed in any category (search stays
global), one match scopes the search to that category, and several
matches scope it to all of them together.
"""

import math

# Cosine similarity a query must clear against a category's name for
# that category to count as matched. Lower than topic_matcher's
# DEFAULT_SIMILARITY_THRESHOLD (0.5) since a category name (e.g.
# "Funding schemes") is a much broader, less specific phrase than a
# topic name — an untuned starting point, not derived from real
# (question, category) pairs the way min_reranker_score/
# topic_match_threshold were; revisit once there's evaluation data to
# tune it against.
DEFAULT_CATEGORY_SIMILARITY_THRESHOLD = 0.35


def _cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class EmbeddingCategoryMatcher:
    """Finds every category (from a fixed list of category names)
    whose name is similar enough to a query to count as matched.

    Category name embeddings are computed once, up front, in a single
    batched embed_texts() call, so matching a query only needs to embed
    the query itself.
    """

    def __init__(
        self,
        categories: list,
        embedding_generator,
        similarity_threshold: float = DEFAULT_CATEGORY_SIMILARITY_THRESHOLD,
    ):
        self._categories = categories
        self._embedding_generator = embedding_generator
        self._similarity_threshold = similarity_threshold
        self._category_embeddings = embedding_generator.embed_texts(categories) if categories else []

    def match(self, query: str, query_embedding: list = None) -> list:
        """Returns every category name clearing similarity_threshold
        against query, most similar first — empty if none do (or there
        are no categories at all), meaning the caller should fall back
        to an unscoped global search.

        query_embedding, if given, is used instead of re-embedding
        query (same reuse pattern as EmbeddingTopicMatcher.match).
        """
        if not self._categories:
            return []
        if query_embedding is None:
            query_embedding = self._embedding_generator.embed_query(query)

        scored = [
            (category, _cosine_similarity(query_embedding, embedding))
            for category, embedding in zip(self._categories, self._category_embeddings)
        ]
        matched = [(category, score) for category, score in scored if score >= self._similarity_threshold]
        matched.sort(key=lambda pair: pair[1], reverse=True)
        return [category for category, _ in matched]
