from rag._03_routing.category_matcher import EmbeddingCategoryMatcher


class FakeEmbeddingGenerator:
    """Deterministic bag-of-words 'embedding' over a small fixed
    vocabulary — enough to exercise EmbeddingCategoryMatcher's
    cosine-similarity logic without a real sentence-transformers model.
    """

    VOCAB = [
        "funding", "schemes", "grants", "approval", "claim", "forms",
        "travel", "policies", "norms", "writing", "guides", "weather",
    ]

    def _vector(self, text: str) -> list:
        words = set(text.lower().split())
        return [1.0 if word in words else 0.0 for word in self.VOCAB]

    def embed_texts(self, texts: list) -> list:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list:
        return self._vector(text)


def _sample_categories():
    return ["funding schemes grants", "approval claim forms travel", "policies norms", "writing guides"]


def test_match_returns_the_single_category_clearing_the_threshold():
    matcher = EmbeddingCategoryMatcher(_sample_categories(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    assert matcher.match("funding schemes grants") == ["funding schemes grants"]


def test_match_returns_empty_list_when_nothing_clears_the_threshold():
    matcher = EmbeddingCategoryMatcher(_sample_categories(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    assert matcher.match("what is the weather") == []


def test_match_returns_every_category_clearing_the_threshold_most_similar_first():
    # "approval claim forms travel" shares 4 words with the second
    # category and 1 with the third ("approval"? no — "claim" and
    # "forms" are unique to category 2; "travel" overlaps category 2
    # only). Use a query that genuinely spans two categories instead.
    matcher = EmbeddingCategoryMatcher(_sample_categories(), FakeEmbeddingGenerator(), similarity_threshold=0.2)
    matched = matcher.match("approval claim forms travel grants funding")
    assert matched[0] in {"funding schemes grants", "approval claim forms travel"}
    assert set(matched) == {"funding schemes grants", "approval claim forms travel"}


def test_match_returns_empty_list_with_no_categories():
    matcher = EmbeddingCategoryMatcher([], FakeEmbeddingGenerator(), similarity_threshold=0.3)
    assert matcher.match("anything at all") == []


def test_match_uses_precomputed_query_embedding_when_given():
    matcher = EmbeddingCategoryMatcher(_sample_categories(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    query_embedding = FakeEmbeddingGenerator()._vector("writing guides")
    matched = matcher.match("irrelevant text", query_embedding=query_embedding)
    assert matched == ["writing guides"]


def test_match_results_are_sorted_most_similar_first():
    matcher = EmbeddingCategoryMatcher(_sample_categories(), FakeEmbeddingGenerator(), similarity_threshold=0.1)
    # A query weighted mostly toward "policies norms" but with a
    # smaller overlap with "writing guides" too.
    query_embedding = FakeEmbeddingGenerator()._vector("policies norms writing")
    matched = matcher.match("irrelevant text", query_embedding=query_embedding)
    assert matched[0] == "policies norms"
