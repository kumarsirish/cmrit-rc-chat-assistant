"""CMRIT-specific retrieval orchestration.

This is the only retrieval module that knows about topics, mapped
documents, or the MAPPED_DOCUMENT/PRIMARY/SECONDARY/GLOBAL policy.
dense_retriever.py, bm25_retriever.py, hybrid_retriever.py, and
reranker.py are generic retrieval mechanisms with no CMRIT knowledge;
this module calls them and decides what to do with the results.

Two retrieval modes, chosen per query:

1. MAPPED_DOCUMENT — the query matches a topic in docs-metadata.json
   that has a locally-indexed document (see routing.topic_matcher.
   EmbeddingTopicMatcher). That topic -> document mapping is trusted,
   so the mapped document is searched first: dense + BM25 search both
   run scoped to that document (filter={"source_file": ...}), fused,
   reranked, filtered by relevance, and kept as up to
   MAPPED_DOCUMENT_PRIMARY_TOP_K "PRIMARY" chunks. If nothing in the
   mapped document clears the relevance threshold, retrieval falls
   back to GLOBAL mode. Otherwise, a second hybrid search runs over
   every *other* document to fill any top_k slots PRIMARY left as
   "SECONDARY" chunks. PRIMARY chunks always precede SECONDARY chunks.
2. GLOBAL — no topic matched, or MAPPED_DOCUMENT mode came up empty.
   Dense + BM25 search run over the whole collection.

Relevance filtering happens before the PRIMARY/SECONDARY allocation
cap, never after: a weaker candidate that happens to fit inside the
cap must not push out a stronger one that would have too. See
_filter_relevant and how PRIMARY/SECONDARY truncate its result.
"""

import logging
import operator
import time
from dataclasses import dataclass

from rag.retrieval import hybrid_retriever
from rag.routing.topic_matcher import DEFAULT_SIMILARITY_THRESHOLD, EmbeddingTopicMatcher

logger = logging.getLogger(__name__)

# Carries the full per-query retrieval trace, separate from `logger`
# above which only logs one summary line per query. Only noisy when a
# DEBUG handler is attached (RAG_DEBUG=true, see dashboard.py).
TRACE_LOGGER_NAME = "rag.retrieval_trace"
_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

# Candidate pool size for MAPPED_DOCUMENT mode. Smaller than GLOBAL
# mode's hybrid_candidates because the search space is already one
# known-correct document.
MAPPED_DOCUMENT_CANDIDATES = 6

# Default cap on how many of the mapped document's own chunks
# ("PRIMARY") are kept, out of top_k, before remaining slots go to a
# SECONDARY, cross-document search. Kept small rather than top_k
# itself: a short, single-subject policy PDF often has several chunks
# that individually clear the relevance threshold, which would
# otherwise let PRIMARY consume the whole top_k budget every time.
MAPPED_DOCUMENT_PRIMARY_TOP_K = 2

MAPPED_DOCUMENT_MODE = "MAPPED_DOCUMENT"
GLOBAL_MODE = "GLOBAL"


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    relevance_score: float
    # "dense" (embedding only), "lexical" (BM25 only), or "hybrid"
    # (found by both). MAPPED_DOCUMENT mode chunks are always "dense"
    # or "hybrid".
    match_type: str = "dense"
    source_link: str = ""
    page_end: int = None
    section_title: str = ""
    chunk_index: int = None
    # True only for the single best-scoring candidate returned as a
    # last-resort fallback when nothing cleared min_reranker_score —
    # see _filter_relevant_by_reranker. Callers (rag_pipeline,
    # dashboard) use this to warn the user the match is weak instead
    # of presenting it with the same confidence as a normal result.
    low_confidence: bool = False

    def __post_init__(self):
        if self.page_end is None:
            self.page_end = self.page_number


def _make_chunk(
    candidate: "hybrid_retriever.FusedCandidate",
    relevance_score: float,
    match_type: str,
    low_confidence: bool = False,
) -> RetrievedChunk:
    metadata = candidate.metadata
    return RetrievedChunk(
        text=candidate.text,
        source_file=metadata.get("source_file"),
        page_number=metadata.get("page_number"),
        relevance_score=relevance_score,
        match_type=match_type,
        source_link=metadata.get("source_link", ""),
        page_end=metadata.get("page_end"),
        section_title=metadata.get("section_title", ""),
        chunk_index=metadata.get("chunk_index"),
        low_confidence=low_confidence,
    )


def _chunk_from_candidate(
    candidate: "hybrid_retriever.FusedCandidate", score: float, low_confidence: bool = False
) -> RetrievedChunk:
    """_make_chunk, deriving match_type ("hybrid"/"lexical"/"dense")
    from which of dense_score/lexical_score the candidate carries."""
    match_type = "hybrid" if candidate.dense_score is not None and candidate.lexical_score is not None else (
        "lexical" if candidate.dense_score is None else "dense"
    )
    return _make_chunk(candidate, score, match_type, low_confidence=low_confidence)


class RetrievalService:
    """Finds the policy chunks most relevant to a user question.

    Retrieves from the topic's mapped document only when
    docs-metadata.json confidently maps the question to one; otherwise
    falls back to hybrid (dense + BM25) search across the whole
    collection.
    """

    def __init__(
        self,
        dense_retriever,
        embedding_generator,
        top_k: int,
        min_reranker_score: float,
        min_relevance_score: float = 0.40,
        bm25_index=None,
        hybrid_candidates: int = 15,
        dense_weight: float = 0.5,
        lexical_weight: float = 0.5,
        reranker=None,
        rerank_candidates: int = 0,
        doc_topics: list = None,
        mapped_document_primary_top_k: int = MAPPED_DOCUMENT_PRIMARY_TOP_K,
        topic_match_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        low_confidence_min_score: float = 0.05,
    ):
        self._dense_retriever = dense_retriever
        self._embedding_generator = embedding_generator
        self._top_k = top_k
        self._min_reranker_score = min_reranker_score
        # Only used for PRIMARY (MAPPED_DOCUMENT) search — see
        # _search's trusted_pool argument. The mapped document is
        # already trusted by the topic match, so PRIMARY relevance is
        # decided from the dense score, same as before reranker_score
        # became the gate for SECONDARY/GLOBAL. This is on the dense
        # cosine [0, 1] scale, not the reranker's scale.
        self._min_relevance_score = min_relevance_score
        # bm25_index is None when lexical retrieval isn't wired up;
        # GLOBAL mode then degrades to dense-only.
        self._bm25_index = bm25_index
        self._hybrid_candidates = hybrid_candidates
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight
        # reranker is None when config.retrieval.rerank.enabled is
        # false; relevance is then decided from the dense score
        # instead of a reranker_score.
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._low_confidence_min_score = low_confidence_min_score
        self._mapped_document_primary_top_k = mapped_document_primary_top_k
        self._topic_matcher = EmbeddingTopicMatcher(
            doc_topics or [], embedding_generator, similarity_threshold=topic_match_threshold
        )

    def match_topic(self, query: str, query_embedding: list = None):
        """Returns the DocTopic query is about, or None if none clears
        the similarity threshold."""
        return self._topic_matcher.match(query, query_embedding=query_embedding)

    def retrieve(
        self,
        query: str,
        metadata_filter: dict = None,
        top_k: int = None,
        min_reranker_score: float = None,
    ) -> tuple:
        """Returns (chunks, matched_topic) for query, chunks most
        relevant first.

        top_k / min_reranker_score, if given, override this service's
        configured defaults for this call only (used by evaluation
        scripts). matched_topic is returned here
        (instead of requiring a second match_topic() call) so a caller
        that also needs it, like rag_pipeline.py, doesn't re-embed and
        re-match the same query a second time.

        Retrieval always runs on query alone — no conversation history
        is ever considered here.

        Flow:
        1. Match the query to a topic. If it maps to a document
           (MAPPED_DOCUMENT mode), search that document alone for up
           to `primary_keep` PRIMARY chunks.
        2. If PRIMARY comes back empty, the topic match isn't trusted
           for this query — fall back to a GLOBAL search over the
           whole collection instead.
        3. Otherwise, fill whatever's left of `top_k` with a second,
           GLOBAL-ish search that excludes the mapped document
           (SECONDARY), so PRIMARY + SECONDARY together total top_k.
        4. No topic match at all skips straight to GLOBAL.
        """
        top_k = self._top_k if top_k is None else top_k
        min_reranker_score = self._min_reranker_score if min_reranker_score is None else min_reranker_score

        overall_start = time.perf_counter()

        query_embedding = self._embedding_generator.embed_query(query)
        matched_topic = self.match_topic(query, query_embedding=query_embedding)
        primary_results = None
        secondary_results = None

        if metadata_filter is None and matched_topic and matched_topic.docs_filename:
            mode = MAPPED_DOCUMENT_MODE
            mapped_documents = matched_topic.docs_filename
            scoped_filter = {"source_file": {"$in": mapped_documents}}
            primary_keep = min(self._mapped_document_primary_top_k, top_k)
            primary_candidates, primary_results = self._search(
                query, query_embedding, scoped_filter, MAPPED_DOCUMENT_CANDIDATES,
                keep=primary_keep, min_reranker_score=min_reranker_score, trusted_pool=True,
            )
            if not primary_results:
                # Mapped document had nothing usable — fall back
                # rather than trusting a possibly-wrong topic match.
                mode = GLOBAL_MODE
                primary_results = None
                candidates, results = self._search(
                    query, query_embedding, metadata_filter, self._hybrid_candidates,
                    keep=top_k, min_reranker_score=min_reranker_score,
                )
            else:
                secondary_keep = top_k - len(primary_results)
                if secondary_keep > 0:
                    secondary_candidates, secondary_results = self._search(
                        query, query_embedding, metadata_filter, self._hybrid_candidates,
                        keep=secondary_keep, min_reranker_score=min_reranker_score,
                        exclude_source_files=mapped_documents,
                    )
                else:
                    secondary_candidates, secondary_results = [], []
                candidates = primary_candidates + secondary_candidates
                results = primary_results + secondary_results
        else:
            mode = GLOBAL_MODE
            candidates, results = self._search(
                query, query_embedding, metadata_filter, self._hybrid_candidates,
                keep=top_k, min_reranker_score=min_reranker_score,
            )

        total_seconds = time.perf_counter() - overall_start
        self._log_summary(
            query, matched_topic, mode, candidates, results, total_seconds,
            primary_results=primary_results, secondary_results=secondary_results,
        )
        return results, matched_topic

    def _search(
        self,
        query: str,
        query_embedding: list,
        metadata_filter: dict,
        candidate_count: int,
        keep: int,
        min_reranker_score: float,
        exclude_source_files: list = None,
        trusted_pool: bool = False,
    ) -> tuple:
        """Dense + BM25 search restricted to metadata_filter, fused,
        filtered by relevance, then capped to `keep`. Filtering always
        happens before the `keep` cap, never after — a candidate that
        would fail the relevance check can't take a slot away from one
        that passes just because it was scored/reranked first.

        trusted_pool is True only for the PRIMARY (MAPPED_DOCUMENT)
        call: that document is already trusted by the topic match, so
        relevance is decided by dense score, and the reranker (if any)
        only reorders that already-relevant pool — it can reorder
        PRIMARY results but can't drop one outright. This matters
        because a cross-encoder can genuinely score a correct chunk
        low (e.g. a short heading like "Welfare Measures for Staff"),
        and letting that evict the mapped document's own top chunk
        just forces a worse GLOBAL fallback. SECONDARY/GLOBAL pools
        are not from a trusted document, so reranker_score stays the
        hard gate there.
        """
        dense_results = self._dense_retriever.search(query_embedding, candidate_count, filter=metadata_filter)
        lexical_results = (
            self._bm25_index.search(query, candidate_count, metadata_filter=metadata_filter)
            if self._bm25_index
            else []
        )
        fused_candidates = hybrid_retriever.fuse(
            dense_results, lexical_results, dense_weight=self._dense_weight, lexical_weight=self._lexical_weight
        )
        if exclude_source_files:
            fused_candidates = [
                candidate for candidate in fused_candidates
                if candidate.metadata.get("source_file") not in exclude_source_files
            ]
        if not fused_candidates:
            return fused_candidates, []

        if trusted_pool:
            relevant_chunks = self._filter_relevant_by_dense_score(fused_candidates, self._min_relevance_score)
            if self._reranker:
                return fused_candidates, self._reranker.rerank(query, relevant_chunks, keep)
            return fused_candidates, relevant_chunks[:keep]

        relevant_chunks = self._filter_relevant(query, fused_candidates, min_reranker_score)
        return fused_candidates, relevant_chunks[:keep]

    def _filter_relevant(self, query: str, fused_candidates: list, min_reranker_score: float) -> list:
        """Turns fused candidates into RetrievedChunk objects, keeping
        only the ones that clear min_reranker_score, most relevant
        first. This is the full relevant pool — callers cap it to
        `keep` themselves, after this filtering has already run.

        When a reranker is configured (the normal case), every
        candidate is scored and judged by the same threshold,
        including a lexical-only (BM25) hit — there is no longer a
        BM25 exemption, since the reranker_score is a real relevance
        judgement of its own, not a dense-cosine comparison BM25
        scores can't be measured against.

        When no reranker is configured, relevance falls back to the
        dense cosine score, matching the app's original threshold
        behavior: a pure dense match below the threshold is dropped,
        but a lexical or hybrid hit is kept (BM25 alone can't be
        compared to that scale, so it isn't thresholded).
        """
        if self._reranker:
            return self._filter_relevant_by_reranker(query, fused_candidates, min_reranker_score)
        return self._filter_relevant_by_dense_score(fused_candidates, min_reranker_score)

    def _filter_relevant_by_reranker(self, query: str, fused_candidates: list, min_reranker_score: float) -> list:
        pool = fused_candidates[: self._rerank_candidates]
        scores = self._reranker.score(query, pool)
        scored_pairs = list(zip(pool, scores))
        scored_pairs.sort(key=operator.itemgetter(1), reverse=True)

        chunks = []
        for candidate, score in scored_pairs:
            if score < min_reranker_score:
                continue
            chunks.append(_chunk_from_candidate(candidate, float(score)))

        if chunks:
            return chunks

        # Nothing cleared the threshold. Rather than refuse outright,
        # fall back to the single best-scoring candidate — but only if
        # its score is at least low_confidence_min_score, so a truly
        # unrelated question (e.g. "What is the capital of France?",
        # which scores ~0.0000-0.002 against this corpus) still gets a
        # real refusal instead of a low-confidence non-answer. Observed
        # real misses that this fallback is meant to rescue scored
        # 0.06-0.31 — well inside that floor. The single chunk is
        # flagged low_confidence=True so callers (rag_pipeline,
        # dashboard) can warn the user to verify against the source
        # instead of presenting it as a normal answer.
        best_candidate, best_score = scored_pairs[0]
        if best_score < self._low_confidence_min_score:
            return []
        return [_chunk_from_candidate(best_candidate, float(best_score), low_confidence=True)]

    def _filter_relevant_by_dense_score(self, fused_candidates: list, min_reranker_score: float) -> list:
        scored_chunks = []  # (sort_key, RetrievedChunk)
        for candidate in fused_candidates:
            if candidate.dense_score is not None:
                relevance_score = candidate.dense_score
                match_type = "hybrid" if candidate.lexical_score is not None else "dense"
                if match_type == "dense" and relevance_score < min_reranker_score:
                    continue
            else:
                relevance_score = min_reranker_score
                match_type = "lexical"
            sort_key = (relevance_score, candidate.fused_score)
            scored_chunks.append((sort_key, _make_chunk(candidate, relevance_score, match_type)))

        scored_chunks.sort(key=operator.itemgetter(0), reverse=True)
        return [chunk for _, chunk in scored_chunks]

    def _log_summary(
        self, query, matched_topic, mode, candidates, results, total_seconds,
        primary_results=None, secondary_results=None,
    ) -> None:
        """Always-on, one-line-per-query summary: query, detected
        topic, mapped document, retrieval mode, candidate/result
        counts, and final source documents."""
        sources = sorted({chunk.source_file for chunk in results})
        logger.info(
            "query=%r topic=%r mapped_document=%r mode=%s candidates=%d final_chunks=%d "
            "sources=%s seconds=%.3f",
            query,
            matched_topic.topic if matched_topic else None,
            matched_topic.docs_filename if matched_topic else None,
            mode,
            len(candidates),
            len(results),
            sources,
            total_seconds,
        )
        if primary_results is not None:
            logger.info(
                "query=%r primary_sources=%s secondary_sources=%s",
                query,
                [chunk.source_file for chunk in primary_results],
                [chunk.source_file for chunk in (secondary_results or [])],
            )
        if _trace_logger.isEnabledFor(logging.DEBUG):
            self._log_debug_trace(query, matched_topic, mode, candidates, results)

    def _log_debug_trace(self, query, matched_topic, mode, candidates, final_chunks) -> None:
        """Writes a fuller per-query retrieval trace, only when a
        DEBUG handler is attached (RAG_DEBUG=true)."""
        lines = [f"QUERY: {query!r}"]
        lines.append(f"1. DETECTED TOPIC: {matched_topic.topic if matched_topic else None}")
        docs_filename = matched_topic.docs_filename if matched_topic else None
        lines.append(f"2. TOPIC -> DOCUMENT MAPPING: {docs_filename!r}")
        lines.append(f"3. RETRIEVAL MODE: {mode}")

        lines.append("4. FUSED CANDIDATES (dense_rank/lexical_rank, pre-relevance-filter):")
        if candidates:
            for candidate in candidates[: self._rerank_candidates or self._top_k]:
                lines.append(
                    f"    dense_rank={candidate.dense_rank} lexical_rank={candidate.lexical_rank} "
                    f"fused={candidate.fused_score:.5f} source={candidate.metadata.get('source_file')!r}"
                )
        else:
            lines.append("    (none)")

        lines.append("5. FINAL CHUNKS SENT TO LLM + SOURCE METADATA:")
        if final_chunks:
            for rank, chunk in enumerate(final_chunks, start=1):
                lines.append(
                    f"    [{rank}] source={chunk.source_file!r} page={chunk.page_number}-{chunk.page_end} "
                    f"relevance={chunk.relevance_score:.4f} type={chunk.match_type} "
                    f"section={chunk.section_title!r} link={chunk.source_link!r}"
                )
        else:
            lines.append("    (none — no answer will be grounded)")

        _trace_logger.debug("\n".join(lines))
