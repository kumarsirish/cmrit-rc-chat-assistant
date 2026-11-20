from rag._05_generation.similar_question_finder import find_similar_question


class FakeEmbeddingGenerator:
    """Deterministic bag-of-words 'embedding' over a small fixed
    vocabulary — enough to exercise cosine-similarity logic (best
    match, threshold rejection) without a real sentence-transformers
    model in unit tests."""

    VOCAB = ["leave", "casual", "sick", "days", "policy", "capital", "france", "weather"]

    def _vector(self, text: str) -> list:
        words = set(text.lower().split())
        return [1.0 if word in words else 0.0 for word in self.VOCAB]

    def embed_query(self, text: str) -> list:
        return self._vector(text)

    def embed_texts(self, texts: list) -> list:
        return [self._vector(text) for text in texts]


def test_find_similar_question_returns_the_closest_match_above_threshold():
    candidates = ["How many days of sick leave do I get?", "What is the capital of France?"]
    result = find_similar_question(
        "How many days of casual leave am I allowed",
        candidates,
        FakeEmbeddingGenerator(),
        similarity_threshold=0.3,
    )
    assert result == "How many days of sick leave do I get?"


def test_find_similar_question_returns_none_when_nothing_clears_the_threshold():
    candidates = ["What is the capital of France?"]
    result = find_similar_question(
        "How many days of casual leave am I allowed",
        candidates,
        FakeEmbeddingGenerator(),
        similarity_threshold=0.3,
    )
    assert result is None


def test_find_similar_question_returns_none_with_no_candidates():
    assert find_similar_question("Any question", [], FakeEmbeddingGenerator()) is None


def test_find_similar_question_ignores_the_exact_same_question():
    # Asking the user "did you mean the exact same question?" would be
    # useless — the only usable candidate is excluded, so nothing matches.
    candidates = ["How many days of casual leave am I allowed"]
    result = find_similar_question(
        "How many days of casual leave am I allowed",
        candidates,
        FakeEmbeddingGenerator(),
        similarity_threshold=0.3,
    )
    assert result is None


def test_find_similar_question_matches_case_and_whitespace_insensitively_for_self_exclusion():
    candidates = ["  HOW MANY DAYS OF CASUAL LEAVE AM I ALLOWED  "]
    result = find_similar_question(
        "how many days of casual leave am i allowed",
        candidates,
        FakeEmbeddingGenerator(),
        similarity_threshold=0.3,
    )
    assert result is None
