"""Step 4: Retrieval.

Two retrieval modes, chosen per query:

1. MAPPED_DOCUMENT — the query matches a topic in docs-metadata.json
   that has a locally-indexed document (see docs_metadata.
   find_matching_topic). That topic -> document mapping is trusted
   metadata, so the mapped document is searched first and given
   priority: dense (embedding) search and BM25 (lexical) search both
   run with a `source_file` filter for just that document, fused by
   RRF over a small candidate pool, optionally narrowed further by the
   cross-encoder reranker, and kept as up to MAPPED_DOCUMENT_PRIMARY_
   TOP_K "PRIMARY" chunks — deliberately less than the full top_k
   budget, so a single-subject document with several individually-
   relevant chunks can't silently consume every slot itself and leave
   none for a second document. Both search methods are kept (not
   dense-only) because the same acronym/proper-noun gap dense search
   has across the whole collection is just as real *within* one
   document — e.g. an acronym like "TYL" is a strong BM25 hit but a
   weak dense one even restricted to its own source document. If
   nothing in the mapped document clears the relevance threshold,
   retrieval falls back to GLOBAL mode rather than trusting a
   possibly-wrong topic match into a dead end. Otherwise, a second,
   ordinary hybrid search runs over every *other* document (the mapped
   one excluded) to fill whatever top_k slots PRIMARY left as
   "SECONDARY" chunks. This is what lets a question whose answer
   legitimately spans more than one document (e.g. "faculty mentoring
   guidelines", covered by two separate policy PDFs) surface the
   second document too, instead of being pinned to only the first one
   docs-metadata.json happened to map it to. PRIMARY chunks always
   precede SECONDARY chunks in the result — a secondary document can
   fill leftover room but can never displace or outrank a primary one,
   no matter its raw score.
2. GLOBAL — no topic matched, the matched topic has no local document
   (only an external link), or MAPPED_DOCUMENT mode came up empty.
   Dense (embedding) search and BM25 (lexical) search run over the
   whole collection, get combined by Reciprocal Rank Fusion (RRF), and
   are optionally narrowed further by the reranker. Dense search alone
   misses exact keyword/name/code matches (acronyms, proper nouns,
   policy codes) that the embedding model has too little signal to
   place correctly; BM25 alone misses paraphrases and synonyms a dense
   embedding captures naturally. Fusing by RANK rather than picking a
   winner or normalizing scores onto a shared scale lets whichever
   signal is actually strong for a given query carry the result,
   without the app having to guess in advance which one that is.

SCORES — three different numbers appear below; they are not on the
same scale and must never be compared to each other directly:
- dense_score: cosine similarity from vector_store, [0, 1], 1 = exact
  match. Used for the min_relevance_score threshold.
- lexical_score: raw BM25 score from bm25_index, unbounded (can be
  negative — see bm25_index.py). Never thresholded, never shown.
- fused_score: Reciprocal Rank Fusion sum, small and rank-derived (see
  RRF_K below). Used only as a tiebreaker within a relevance_score
  group; never shown to the user. Computed the same way in both modes
  — in MAPPED_DOCUMENT mode it only ever ranks chunks within the one
  mapped document, since that's all fusion ever sees there.
- relevance_score (on RetrievedChunk): the score the rest of the app
  sees, AND the primary key final ordering is sorted by. Equals
  dense_score when there is one, else a conservative
  min_relevance_score sentinel for a lexical-only hit (BM25 finding a
  chunk means it matched lexically, not that it's highly relevant; the
  reranker is what actually judges that). Also used for the
  min_relevance_score threshold and shown in the UI/debug panel.
"""

import logging
import operator
import time
from dataclasses import dataclass

from rag.indexing.docs_metadata import find_matching_topic
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Separate from `logger` above (which only ever logs one compact
# per-query summary line — see Retriever._log_summary): this one
# carries the full per-query retrieval trace — see
# Retriever._log_debug_trace — and is only ever noisy when something
# has explicitly attached a handler to it and set it to DEBUG, which
# the dashboard does only under RAG_DEBUG=true (see dashboard.py). No
# handler is attached here, so by default this is a no-op.
TRACE_LOGGER_NAME = "rag.retrieval_trace"
_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

# Reciprocal Rank Fusion constant: fused_score = sum(1 / (RRF_K + rank))
# over every ranked list (dense, lexical) a chunk appears in (rank is
# 0-based). 60 is the constant from the original RRF paper; it damps
# how much a #1-vs-#2 difference matters without needing to be tuned
# per corpus size. Used in both modes — in MAPPED_DOCUMENT mode both
# ranked lists (dense, lexical) are already restricted to one document.
RRF_K = 60

# Candidate pool size for MAPPED_DOCUMENT mode. Deliberately much
# smaller than GLOBAL mode's hybrid_candidates: the search space is
# already restricted to one known-correct document, so there's no
# need to over-fetch the way GLOBAL mode does to give RRF/reranking
# enough to work with across the whole collection.
MAPPED_DOCUMENT_CANDIDATES = 6

# Cap on how many of the mapped document's own chunks ("PRIMARY") are
# kept, out of top_k, before any remaining slots go to a SECONDARY,
# cross-document search. Deliberately small rather than top_k itself:
# a short, single-subject policy PDF frequently has several chunks
# that individually clear min_relevance_score (they're all "about" the
# same narrow topic), which would otherwise let PRIMARY silently
# consume the *entire* top_k budget on its own every time and leave
# SECONDARY no room to ever run — defeating the point of adding it.
# Observed on the real corpus: "Give TYL Program Details" and "What
# guidelines exist for faculty mentoring at CMRIT?" (a genuinely
# multi-document question) both filled all of top_k from PRIMARY alone
# with no cap, so a second, actually-relevant document never surfaced.
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
    # (found by both). MAPPED_DOCUMENT mode chunks are always "dense".
    match_type: str = "dense"
    source_link: str = ""
    page_end: int = None
    section_title: str = ""
    chunk_index: int = None

    def __post_init__(self):
        if self.page_end is None:
            self.page_end = self.page_number


def _chunk_key(metadata: dict) -> tuple:
    return (
        metadata.get("source_file"),
        metadata.get("page_number"),
        metadata.get("chunk_index", 0),
    )


@dataclass
class _FusedCandidate:
    text: str
    metadata: dict
    dense_score: float = None
    lexical_score: float = None
    fused_score: float = 0.0


def _format_search_hit(document, score) -> str:
    return f"    score={score:.4f} source={document.metadata.get('source_file')!r} page={document.metadata.get('page_number')}"


def _format_fused_candidate(candidate: "_FusedCandidate") -> str:
    return (
        f"    fused={candidate.fused_score:.5f} dense={candidate.dense_score} "
        f"lexical={candidate.lexical_score} source={candidate.metadata.get('source_file')!r}"
    )


def _format_chunk(chunk: "RetrievedChunk") -> str:
    return (
        f"    relevance={chunk.relevance_score:.4f} type={chunk.match_type} "
        f"source={chunk.source_file!r} page={chunk.page_number}"
    )


def _document_text(document) -> str:
    # Dense results are langchain Documents (.page_content); BM25
    # results are BM25Document (.text, see bm25_index.py).
    return document.page_content if hasattr(document, "page_content") else document.text


def _make_chunk(text: str, metadata: dict, relevance_score: float, match_type: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_file=metadata.get("source_file"),
        page_number=metadata.get("page_number"),
        relevance_score=relevance_score,
        match_type=match_type,
        source_link=metadata.get("source_link", ""),
        page_end=metadata.get("page_end"),
        section_title=metadata.get("section_title", ""),
        chunk_index=metadata.get("chunk_index"),
    )


def reciprocal_rank_fusion(dense_results: list, lexical_results: list) -> list:
    """Combines dense_results and lexical_results — each a list of
    (document, score) most-relevant-first — into one list of
    _FusedCandidate ranked by RRF score (descending). Used by both
    MAPPED_DOCUMENT mode (dense_results/lexical_results already
    restricted to one document) and GLOBAL mode (unrestricted, or
    restricted to a caller-given metadata_filter).

    RRF combines by rank rather than raw score, so cosine similarity
    and BM25 scores (entirely different scales) can be merged without
    normalizing either one onto a common range.
    """
    candidates = {}

    def _accumulate(results: list, score_attr: str) -> None:
        for rank, (document, score) in enumerate(results):
            key = _chunk_key(document.metadata)
            candidate = candidates.get(key)
            if candidate is None:
                candidate = _FusedCandidate(text=_document_text(document), metadata=document.metadata)
                candidates[key] = candidate
            setattr(candidate, score_attr, score)
            candidate.fused_score += 1.0 / (RRF_K + rank + 1)

    _accumulate(dense_results, "dense_score")
    _accumulate(lexical_results, "lexical_score")

    return sorted(candidates.values(), key=operator.attrgetter("fused_score"), reverse=True)


class Retriever:
    """Finds the policy chunks most relevant to a user question.

    Retrieves from the topic's mapped document only when
    docs-metadata.json confidently maps the question to one (see
    MAPPED_DOCUMENT_MODE); otherwise falls back to hybrid (dense +
    BM25) search across the whole collection (GLOBAL_MODE)."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        top_k: int,
        min_relevance_score: float,
        bm25_index=None,
        hybrid_candidates: int = 15,
        reranker=None,
        rerank_candidates: int = 0,
        doc_topics: list = None,
    ):
        self._vector_store = vector_store
        self._embedding_generator = embedding_generator
        self._top_k = top_k
        self._min_relevance_score = min_relevance_score
        # bm25_index is None when lexical retrieval isn't wired up
        # (e.g. some unit tests); GLOBAL mode then degrades to
        # dense-only.
        self._bm25_index = bm25_index
        self._hybrid_candidates = hybrid_candidates
        # reranker is None when config.retrieval.rerank.enabled is
        # false; the candidate pool is then truncated to top_k
        # directly by its existing order instead of being re-scored.
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._doc_topics = doc_topics or []

    def retrieve(self, query: str, metadata_filter: dict = None) -> list:
        """Returns relevant chunks for query, most relevant first.

        Retrieval always runs on query alone — no conversation history
        is ever considered here; callers that want follow-up context
        resolved pass it to the LLM separately (see rag_pipeline.py).

        metadata_filter, if given, forces GLOBAL-mode hybrid search
        restricted to matching chunks (e.g. {"source_file": "leave.pdf"})
        and takes priority over topic detection — used by callers that
        already know exactly which document(s) to search. When it's
        not given, a query that matches a topic with a locally-indexed
        document runs MAPPED_DOCUMENT mode: a scoped search against
        that document alone produces up to MAPPED_DOCUMENT_PRIMARY_
        TOP_K "PRIMARY" chunks (see _search), which always precede
        every other chunk in the result — nothing (reranker included)
        can reorder a non-primary chunk ahead of one. If that document
        has nothing above the relevance threshold — the topic match
        was wrong, or the document genuinely doesn't answer the
        question — retrieval falls back fully to a normal GLOBAL
        search rather than trusting the mapping blindly into a dead
        end. Otherwise, if PRIMARY left any of top_k unfilled, a
        second hybrid search over every document *except* the mapped
        one fills the remaining slots as "SECONDARY" chunks, so a
        question whose answer genuinely spans more than one document
        isn't pinned to only the one docs-metadata.json mapped it to.
        A query that matches no topic (or a topic with no local
        document, just an external link) runs GLOBAL, hybrid search +
        RRF fusion + optional reranking, over the whole collection.
        """
        overall_start = time.perf_counter()

        query_embedding = self._embedding_generator.embed_query(query)
        matched_topic = find_matching_topic(query, self._doc_topics)
        primary_results = None
        secondary_results = None

        if metadata_filter is None and matched_topic and matched_topic.docs_filename:
            mode = MAPPED_DOCUMENT_MODE
            mapped_document = matched_topic.docs_filename
            scoped_filter = {"source_file": mapped_document}
            # Capped at MAPPED_DOCUMENT_PRIMARY_TOP_K (not the full
            # top_k budget) so PRIMARY can't consume every slot on its
            # own and leave SECONDARY nothing to fill — see that
            # constant's comment.
            primary_keep = min(MAPPED_DOCUMENT_PRIMARY_TOP_K, self._top_k)
            primary_candidates, primary_results = self._search(
                query, query_embedding, scoped_filter, MAPPED_DOCUMENT_CANDIDATES, keep=primary_keep
            )
            if not primary_results:
                # Mapped document had nothing usable — fall back
                # rather than returning an empty result for a query
                # that may just have matched the wrong topic. No
                # primary+secondary merge in this case (see module
                # docstring): a wrong topic match has nothing to be
                # "primary" about.
                mode = GLOBAL_MODE
                primary_results = None
                candidates, results = self._search(query, query_embedding, metadata_filter, self._hybrid_candidates)
            else:
                secondary_keep = self._top_k - len(primary_results)
                if secondary_keep > 0:
                    secondary_candidates, secondary_results = self._search(
                        query,
                        query_embedding,
                        metadata_filter,
                        self._hybrid_candidates,
                        keep=secondary_keep,
                        exclude_source_file=mapped_document,
                    )
                else:
                    secondary_candidates, secondary_results = [], []
                candidates = primary_candidates + secondary_candidates
                results = primary_results + secondary_results
        else:
            mode = GLOBAL_MODE
            candidates, results = self._search(query, query_embedding, metadata_filter, self._hybrid_candidates)

        total_seconds = time.perf_counter() - overall_start
        self._log_summary(
            query,
            matched_topic,
            mode,
            candidates,
            results,
            total_seconds,
            primary_results=primary_results,
            secondary_results=secondary_results,
        )
        return results

    def _search(
        self,
        query: str,
        query_embedding: list,
        metadata_filter: dict,
        candidate_count: int,
        keep: int = None,
        exclude_source_file: str = None,
    ) -> tuple:
        """Dense + BM25 search restricted to metadata_filter (None for
        the whole collection, {"source_file": ...} for one document),
        fused by RRF, scored/filtered, then reranked/truncated to
        `keep` (self._top_k if not given). The same retrieval
        primitives serve MAPPED_DOCUMENT mode's PRIMARY search
        (metadata_filter pins one document, candidate_count is small,
        keep is MAPPED_DOCUMENT_PRIMARY_TOP_K), MAPPED_DOCUMENT mode's
        SECONDARY search (metadata_filter is
        None, exclude_source_file drops the mapped document's own
        chunks from the pool, keep is whatever top_k slots PRIMARY
        left unfilled), and GLOBAL mode (metadata_filter is None or
        caller-given, candidate_count is hybrid_candidates, keep is
        self._top_k) — only the search scope, exclusion, and pool/keep
        sizes differ; when metadata_filter pins one document, fusion
        and reranking can only ever reorder chunks within it.
        """
        keep = self._top_k if keep is None else keep
        dense_results = self._vector_store.similarity_search_with_relevance_scores(
            query_embedding, candidate_count, filter=metadata_filter
        )
        lexical_results = (
            self._bm25_index.search(query, candidate_count, metadata_filter=metadata_filter)
            if self._bm25_index
            else []
        )
        fused_candidates = reciprocal_rank_fusion(dense_results, lexical_results)
        candidates = self._score_and_filter(fused_candidates)
        if exclude_source_file:
            candidates = [candidate for candidate in candidates if candidate.source_file != exclude_source_file]
        if not candidates:
            return candidates, []
        return candidates, self._rerank_or_truncate(query, candidates, keep)

    def _score_and_filter(self, fused_candidates: list) -> list:
        """Turns RRF-fused candidates into RetrievedChunk objects,
        deciding each one's relevance_score/match_type, dropping weak
        unconfirmed matches, and ordering the result. Used by both
        modes (see _search)."""
        scored_chunks = []  # ((relevance_score, fused_score), RetrievedChunk)
        for candidate in fused_candidates:
            if candidate.dense_score is not None:
                relevance_score = candidate.dense_score
                match_type = "hybrid" if candidate.lexical_score is not None else "dense"
                if match_type == "dense" and relevance_score < self._min_relevance_score:
                    # Only a pure, unconfirmed dense match is dropped
                    # for weak similarity; a lexical hit or a
                    # dense+lexical (hybrid) agreement is corroborated
                    # by an independent signal and is kept regardless.
                    continue
            else:
                # Lexical-only hit: BM25 scores aren't on the [0, 1]
                # cosine scale min_relevance_score assumes, so a
                # literal keyword match found by BM25 skips the
                # threshold rather than being scored against it. Its
                # relevance_score is set to the threshold itself — a
                # neutral "found, unconfirmed" baseline — rather than a
                # false 1.0 sentinel; BM25 only proves lexical overlap,
                # not that the chunk is highly relevant, and it's the
                # reranker/downstream relevance logic, not this score,
                # that decides whether it's actually useful.
                relevance_score = self._min_relevance_score
                match_type = "lexical"

            scored_chunks.append(
                (
                    (relevance_score, candidate.fused_score),
                    _make_chunk(candidate.text, candidate.metadata, relevance_score, match_type),
                )
            )

        scored_chunks.sort(key=operator.itemgetter(0), reverse=True)
        return [chunk for _, chunk in scored_chunks]

    def _rerank_or_truncate(self, query: str, candidates: list, keep: int) -> list:
        """Narrows candidates to `keep`: via the cross-encoder reranker
        if one is configured, or by simple truncation (already in rank
        order) if not. When candidates is a MAPPED_DOCUMENT PRIMARY
        pool, every candidate already belongs to the mapped document,
        so reranking can only reorder within it — it can never pull
        in, or be replaced by, a different document; a SECONDARY pool
        is reranked/truncated the same way, just over every other
        document instead."""
        if not candidates:
            return candidates
        if self._reranker:
            return self._reranker.rerank(query, candidates[: self._rerank_candidates], keep)
        return candidates[:keep]

    def _log_summary(
        self,
        query: str,
        matched_topic,
        mode: str,
        candidates: list,
        results: list,
        total_seconds: float,
        primary_results: list = None,
        secondary_results: list = None,
    ) -> None:
        """Always-on, one-line-per-query summary (see module docstring
        item 12 in the design this implements): query, detected topic,
        mapped document, retrieval mode, candidate/result counts, and
        final source documents — enough to diagnose a retrieval
        failure without needing RAG_DEBUG's full trace. For a
        successful MAPPED_DOCUMENT query, also logs the PRIMARY (mapped
        document) vs SECONDARY (everything else) source-file split, so
        a multi-document answer's provenance is visible without
        RAG_DEBUG's full trace either."""
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

    def _log_debug_trace(
        self,
        query: str,
        matched_topic,
        mode: str,
        candidates: list,
        final_chunks: list,
    ) -> None:
        """Writes a fuller per-query retrieval trace to
        TRACE_LOGGER_NAME — only ever called when that logger has a
        DEBUG-level handler attached (RAG_DEBUG=true, see
        dashboard.py)."""
        lines = [f"QUERY: {query!r}"]

        lines.append(f"1. DETECTED TOPIC: {matched_topic.topic if matched_topic else None}")
        docs_filename = matched_topic.docs_filename if matched_topic else None
        lines.append(f"2. TOPIC -> DOCUMENT MAPPING: {docs_filename!r}")
        lines.append(f"3. RETRIEVAL MODE: {mode}")

        lines.append("4. CANDIDATES (post scoring/filtering, pre-rerank):")
        if candidates:
            shown = candidates[: self._rerank_candidates] if self._reranker else candidates[: self._top_k]
            lines.extend(_format_chunk(chunk) for chunk in shown)
        else:
            lines.append("    (none passed the relevance threshold)")

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
