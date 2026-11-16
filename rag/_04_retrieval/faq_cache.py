"""Redis-backed cache for frequently-repeated questions' retrieval
results (chunks + matched topic), so a question many different users
ask verbatim — e.g. the dashboard's example quick-questions — skips
embedding + dense/BM25 search + reranking on repeat.

Deliberately narrow:
- Caches only exact question matches (after trivial normalization) —
  no fuzzy/semantic cache-key matching, so a cache hit is always
  provably the same question, never an approximate one that could
  silently serve the wrong chunks for a similar-but-different query.
- Only promotes a question into the cache once it's been asked at
  least faq_cache_min_hits times (see config.yaml's redis section) —
  a first-time question always goes through full retrieval, so the
  cache only ever holds genuine repeat traffic.
- Wiped entirely whenever ingestion completes (see run_ingestion's
  call to reset_if_enabled) — cached chunks/topics are tied to the
  vector index's exact contents, and a re-ingest can add, remove, or
  renumber chunks, so a stale entry could otherwise serve content
  that's no longer indexed the same way.
- Any Redis failure (unreachable, timeout, ...) falls back to calling
  the wrapped RetrievalService directly — this is a performance layer,
  never a hard dependency for answering a question.
"""

import dataclasses
import json
import logging
import re

from rag._03_routing.docs_metadata import DocTopic
from rag._04_retrieval.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)

_WHITESPACE_PATTERN = re.compile(r"\s+")
_HITS_KEY = "faq:hits"


def _normalize(question: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", question.strip().lower())


def build_client(redis_config):
    """redis_config is AppConfig.redis. Import is local to this
    function so the `redis` package is only required when the cache is
    actually enabled."""
    import redis

    return redis.Redis(
        host=redis_config.host,
        port=redis_config.port,
        db=redis_config.db,
        password=redis_config.password or None,
        decode_responses=True,
    )


def maybe_wrap(retrieval_service, redis_config):
    """Returns retrieval_service wrapped in CachedRetrievalService if
    redis_config.enabled, otherwise returns it unchanged."""
    if not redis_config.enabled:
        return retrieval_service
    return CachedRetrievalService(retrieval_service, build_client(redis_config), redis_config)


def reset_if_enabled(redis_config) -> None:
    """Wipes every cached question/chunk and hit-count — call this
    once a fresh ingestion finishes (see helper_scripts/ingest.py),
    since cached chunks are only valid against the index they were
    retrieved from."""
    if not redis_config.enabled:
        return
    build_client(redis_config).flushdb()


def _serialize(chunks: list, matched_topic) -> str:
    return json.dumps(
        {
            "chunks": [dataclasses.asdict(chunk) for chunk in chunks],
            "matched_topic": dataclasses.asdict(matched_topic) if matched_topic else None,
        }
    )


def _deserialize(raw: str) -> tuple:
    payload = json.loads(raw)
    chunks = [RetrievedChunk(**chunk) for chunk in payload["chunks"]]
    matched_topic = DocTopic(**payload["matched_topic"]) if payload["matched_topic"] else None
    return chunks, matched_topic


class CachedRetrievalService:
    """Wraps a RetrievalService, caching retrieve() results for
    questions asked often enough (see module docstring). match_topic()
    passes straight through unchanged — it's cheap (one embedding plus
    cosine comparisons against a handful of topics), not worth caching."""

    def __init__(self, retrieval_service, redis_client, redis_config):
        self._retrieval_service = retrieval_service
        self._redis = redis_client
        self._min_hits = redis_config.faq_cache_min_hits
        self._ttl_seconds = redis_config.cache_ttl_seconds

    def match_topic(self, query, query_embedding=None):
        return self._retrieval_service.match_topic(query, query_embedding=query_embedding)

    def retrieve(self, query, metadata_filter=None, top_k=None, min_reranker_score=None):
        # top_k/min_reranker_score overrides are only used by
        # evaluation/experiment scripts to try non-default settings —
        # caching those would either serve a default-settings result
        # under an override, or pollute the cache with results that
        # don't match the app's actual defaults. Bypass the cache
        # entirely for that case.
        if top_k is not None or min_reranker_score is not None:
            return self._retrieval_service.retrieve(
                query, metadata_filter=metadata_filter, top_k=top_k, min_reranker_score=min_reranker_score
            )

        normalized = _normalize(query)
        try:
            hits = self._redis.hincrby(_HITS_KEY, normalized, 1)
            cached = self._redis.get(f"faq:chunks:{normalized}")
        except Exception:
            logger.warning("FAQ cache read failed; falling back to direct retrieval.", exc_info=True)
            return self._retrieval_service.retrieve(query, metadata_filter=metadata_filter)

        if cached is not None:
            return _deserialize(cached)

        chunks, matched_topic = self._retrieval_service.retrieve(query, metadata_filter=metadata_filter)

        if hits >= self._min_hits:
            try:
                self._redis.set(
                    f"faq:chunks:{normalized}", _serialize(chunks, matched_topic), ex=self._ttl_seconds
                )
            except Exception:
                logger.warning("FAQ cache write failed; answer is unaffected.", exc_info=True)

        return chunks, matched_topic
