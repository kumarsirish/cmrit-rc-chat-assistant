"""Unit tests for CachedRetrievalService — a fake in-memory Redis
client stands in for the real one so these don't need a live server.
"""

from rag._03_routing.docs_metadata import DocTopic
from rag._04_retrieval import faq_cache
from rag._04_retrieval.retrieval_service import RetrievedChunk


class FakeRedis:
    """Just enough of the redis-py surface for CachedRetrievalService:
    a hash (hincrby) and plain string get/set with expiry ignored."""

    def __init__(self):
        self.hashes = {}
        self.strings = {}

    def hincrby(self, name, key, amount=1):
        bucket = self.hashes.setdefault(name, {})
        bucket[key] = bucket.get(key, 0) + amount
        return bucket[key]

    def get(self, key):
        return self.strings.get(key)

    def set(self, key, value, ex=None):
        self.strings[key] = value

    def flushdb(self):
        self.hashes.clear()
        self.strings.clear()


class FakeRetrievalService:
    def __init__(self, chunks, matched_topic=None):
        self._chunks = chunks
        self._matched_topic = matched_topic
        self.call_count = 0

    def retrieve(self, query, metadata_filter=None, top_k=None, min_reranker_score=None):
        self.call_count += 1
        return self._chunks, self._matched_topic

    def match_topic(self, query, query_embedding=None):
        return self._matched_topic


class FakeRedisConfig:
    def __init__(self, faq_cache_min_hits=3, cache_ttl_seconds=86400, enabled=True):
        self.faq_cache_min_hits = faq_cache_min_hits
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enabled = enabled


def _chunk(**overrides):
    defaults = dict(text="Casual leave is 12 days a year.", source_file="leave.pdf", page_number=4, relevance_score=0.9)
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_below_min_hits_never_caches_and_always_calls_the_real_service():
    inner = FakeRetrievalService(chunks=[_chunk()])
    redis_client = FakeRedis()
    cached = faq_cache.CachedRetrievalService(inner, redis_client, FakeRedisConfig(faq_cache_min_hits=3))

    cached.retrieve("What is the CMRIT quality policy?")
    cached.retrieve("What is the CMRIT quality policy?")

    assert inner.call_count == 2
    assert redis_client.strings == {}


def test_reaching_min_hits_caches_and_stops_calling_the_real_service():
    inner = FakeRetrievalService(chunks=[_chunk()], matched_topic=DocTopic(topic="Quality Policy", link="https://example.com"))
    redis_client = FakeRedis()
    cached = faq_cache.CachedRetrievalService(inner, redis_client, FakeRedisConfig(faq_cache_min_hits=2))

    cached.retrieve("What is the CMRIT quality policy?")
    chunks, topic = cached.retrieve("What is the CMRIT quality policy?")
    chunks_again, topic_again = cached.retrieve("What is the CMRIT quality policy?")

    # Only the first two calls reached the real service; the third was
    # served from cache.
    assert inner.call_count == 2
    assert chunks_again[0].source_file == "leave.pdf"
    assert topic_again.topic == "Quality Policy"


def test_normalization_treats_case_and_whitespace_variants_as_the_same_question():
    inner = FakeRetrievalService(chunks=[_chunk()])
    redis_client = FakeRedis()
    cached = faq_cache.CachedRetrievalService(inner, redis_client, FakeRedisConfig(faq_cache_min_hits=2))

    cached.retrieve("What is the CMRIT quality policy?")
    cached.retrieve("  WHAT IS   THE cmrit quality policy?  ")
    cached.retrieve("what is the cmrit quality policy?")

    assert inner.call_count == 2


def test_top_k_override_bypasses_the_cache_entirely():
    inner = FakeRetrievalService(chunks=[_chunk()])
    redis_client = FakeRedis()
    cached = faq_cache.CachedRetrievalService(inner, redis_client, FakeRedisConfig(faq_cache_min_hits=1))

    cached.retrieve("What is the CMRIT quality policy?", top_k=5)
    cached.retrieve("What is the CMRIT quality policy?", top_k=5)

    assert inner.call_count == 2
    assert redis_client.hashes == {}
    assert redis_client.strings == {}


def test_redis_failure_falls_back_to_the_real_service():
    class BrokenRedis(FakeRedis):
        def hincrby(self, name, key, amount=1):
            raise ConnectionError("redis down")

    inner = FakeRetrievalService(chunks=[_chunk()])
    cached = faq_cache.CachedRetrievalService(inner, BrokenRedis(), FakeRedisConfig())

    chunks, _ = cached.retrieve("What is the CMRIT quality policy?")

    assert inner.call_count == 1
    assert chunks[0].source_file == "leave.pdf"


def test_maybe_wrap_returns_service_unchanged_when_disabled():
    inner = FakeRetrievalService(chunks=[])
    result = faq_cache.maybe_wrap(inner, FakeRedisConfig(enabled=False))
    assert result is inner
