from rag._03_routing.document_metadata import DocTopic
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


def test_match_all_picks_the_most_semantically_similar_topic_first():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    matched = matcher.match_all("vacation leave application")
    assert matched[0].topic == "leave application process approval"


def test_match_all_returns_every_topic_clearing_zero_overlap_threshold():
    # A threshold low enough that multiple sample topics clear it for
    # a query overlapping both "leave"/"application" and "student" —
    # proving match_all CAN return more than one match. similarity_margin
    # is widened to isolate that from the margin behavior tested below,
    # since these two sample topics don't score within the default 0.1
    # margin of each other.
    matcher = EmbeddingTopicMatcher(
        _sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.2, similarity_margin=1.0
    )
    matched = matcher.match_all("leave application student handbook")
    assert {topic.topic for topic in matched} == {
        "leave application process approval",
        "student handbook",
    }


def test_match_all_excludes_a_topic_outside_the_margin_of_the_best_match():
    # Same query/threshold as above, but with the default margin (0.1):
    # "student handbook" scores far higher (~0.71) than "leave
    # application process approval" (~0.5) for this query — outside a
    # 0.1 margin of the best, so only the best survives even though
    # both individually clear the threshold.
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.2)
    matched = matcher.match_all("leave application student handbook")
    assert [topic.topic for topic in matched] == ["student handbook"]


def test_match_all_keeps_topics_within_the_margin_of_the_best_match():
    # "leave application process approval" and a near-duplicate phrase
    # score close enough to each other (within 0.1) that both should
    # survive — this is the genuine multi-topic case match_all exists
    # for (e.g. two related forms in the same category).
    topics = [
        DocTopic(topic="leave application process", link="https://example.com/a", docs_filename="a.pdf"),
        DocTopic(topic="leave application approval", link="https://example.com/b", docs_filename="b.pdf"),
    ]
    matcher = EmbeddingTopicMatcher(topics, FakeEmbeddingGenerator(), similarity_threshold=0.2)
    matched = matcher.match_all("leave application")
    assert {topic.topic for topic in matched} == {"leave application process", "leave application approval"}


def test_match_all_respects_the_max_matches_cap():
    topics = [
        DocTopic(topic=f"leave application {suffix}", link=f"https://example.com/{suffix}", docs_filename=f"{suffix}.pdf")
        for suffix in ("a", "b", "c", "d")
    ]
    matcher = EmbeddingTopicMatcher(
        topics, FakeEmbeddingGenerator(), similarity_threshold=0.2, similarity_margin=1.0, max_matches=2
    )
    matched = matcher.match_all("leave application")
    assert len(matched) == 2


def test_match_all_returns_empty_when_nothing_clears_the_threshold():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    assert matcher.match_all("what is the weather today") == []


def test_match_all_respects_a_higher_similarity_threshold():
    # "leave" overlaps only one of four words in the topic name — a
    # weak match that a low threshold accepts but a strict one rejects.
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.9)
    assert matcher.match_all("leave") == []


def test_match_all_finds_link_only_entry():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    matched = matcher.match_all("website access")
    assert matched[0].topic == "website access portal"
    assert matched[0].docs_filename == []


def test_match_all_returns_empty_with_no_topics():
    matcher = EmbeddingTopicMatcher([], FakeEmbeddingGenerator(), similarity_threshold=0.3)
    assert matcher.match_all("anything at all") == []


def test_match_all_uses_precomputed_query_embedding_when_given():
    matcher = EmbeddingTopicMatcher(_sample_topics(), FakeEmbeddingGenerator(), similarity_threshold=0.3)
    # A query embedding identical to the "student handbook" topic
    # vector, passed in directly — the query text itself ("irrelevant
    # text") would not match anything on its own, proving the given
    # embedding (not a re-embedding of the text) drove the match.
    query_embedding = FakeEmbeddingGenerator()._vector("student handbook")
    matched = matcher.match_all("irrelevant text", query_embedding=query_embedding)
    assert matched[0].topic == "student handbook"
