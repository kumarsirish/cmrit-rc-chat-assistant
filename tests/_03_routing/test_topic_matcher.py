from rag._03_routing.docs_metadata import DocTopic
from rag._03_routing.topic_matcher import EmbeddingTopicMatcher


class FakeEmbeddingGenerator:
    """Deterministic bag-of-words 'embedding' over a small fixed
    vocabulary — enough to exercise EmbeddingTopicMatcher's
    cosine-similarity logic (best match, threshold rejection, ties)
    without needing a real sentence-transformers model in unit tests.
    """

    VOCAB = [
        "leave", "vacation", "application", "process", "approval",
        "website", "access", "portal", "student", "handbook",
        "weather", "today",
    ]

    def _vector(self, text: str) -> list:
        words = set(text.lower().split())
        return [1.0 if word in words else 0.0 for word in self.VOCAB]

    def embed_texts(self, texts: list) -> list:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list:
        return self._vector(text)


def _sample_topics():
    return [
        DocTopic(
            topic="leave application process approval",
            link="https://example.com/leave",
            docs_filename="CMRIT - Leave Information.pdf",
        ),
        DocTopic(
            topic="website access portal",
            link="http://www.cmrit.ac.in",
            docs_filename=None,
        ),
        DocTopic(
            topic="student handbook",
            link="https://example.com/student-handbook",
            docs_filename="Student Handbook-CMRIT.docx.pdf",
        ),
    ]


def test_match_picks_the_most_semantically_similar_topic():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    matched = matcher.match("vacation leave application")
    assert matched.topic == "leave application process approval"


def test_match_returns_none_when_nothing_clears_the_threshold():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    assert matcher.match("what is the weather today") is None


def test_match_respects_a_higher_similarity_threshold():
    # "leave" overlaps only one of four words in the topic name — a
    # weak match that a low threshold accepts but a strict one rejects.
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.9)
    assert matcher.match("leave") is None


def test_match_finds_link_only_entry():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    matched = matcher.match("website access")
    assert matched.topic == "website access portal"
    assert matched.docs_filename == []


def test_match_returns_none_with_no_topics():
    matcher = EmbeddingTopicMatcher([], FakeEmbeddingGenerator(), similarity_threshold=0.3)
    assert matcher.match("anything at all") is None


def test_match_uses_precomputed_query_embedding_when_given():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    # A query embedding identical to the "student handbook" topic
    # vector, passed in directly — the query text itself ("irrelevant
    # text") would not match anything on its own, proving the given
    # embedding (not a re-embedding of the text) drove the match.
    query_embedding = FakeEmbeddingGenerator()._vector("student handbook")
    matched = matcher.match("irrelevant text", query_embedding=query_embedding)
    assert matched.topic == "student handbook"
