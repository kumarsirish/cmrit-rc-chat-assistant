"""Topic-aware retrieval orchestration.

This is the only retrieval module that knows about topics, categories,
or the FOCUSSED_CATEGORY/PRIMARY/SECONDARY/GLOBAL policy.
dense_retriever.py, bm25_retriever.py, hybrid_retriever.py, and
reranker.py are generic retrieval mechanisms with no topic knowledge;
this module calls them and decides what to do with the results.

Two retrieval modes, chosen per query from category_override alone —
there is no automatic category detection:

1. FOCUSSED_CATEGORY — the caller (the UI's category picker) has
   named a specific category. The query is topic-matched against only
   that category's own documents (see topic_matcher.
   EmbeddingTopicMatcher): every topic clearing the threshold counts
   as matched, not just the single best one, since a query can
   legitimately be about more than one document in the same category.
   PRIMARY searches the union of those matched documents; if nothing
   there clears relevance, retrieval falls back to a flat
   category-scoped search instead of trusting a possibly-wrong match.
   Otherwise, a second hybrid search over the rest of that category
   fills any top_k slots PRIMARY left as SECONDARY chunks.
2. GLOBAL — category_override is None or ALL_CATEGORIES. Same
   PRIMARY/SECONDARY structure as FOCUSSED_CATEGORY, except topic-
   matching runs against every active document in the whole corpus
   instead of one category's, and the SECONDARY/fallback searches are
   unscoped.

Relevance filtering happens before the PRIMARY/SECONDARY allocation
cap, never after: a weaker candidate that happens to fit inside the
cap must not push out a stronger one that would have too. See
_filter_relevant and how PRIMARY/SECONDARY truncate its result.
"""

import logging
import operator
import time
from dataclasses import dataclass

from rag._04_retrieval import hybrid_retriever
from rag._03_routing.query_expander import strip_institution_name
from rag._03_routing.topic_matcher import (
    DEFAULT_MAX_MATCHES,
    DEFAULT_SIMILARITY_MARGIN,
    DEFAULT_SIMILARITY_THRESHOLD,
    EmbeddingTopicMatcher,
)

logger = logging.getLogger(__name__)

# Candidate pool size for PRIMARY search. Larger than a single mapped
# document would need, since PRIMARY can now span every document
# whose topic matched (several documents in the same category, not
# just one) — see match_topics.
PRIMARY_CANDIDATES = 12

# Default cap on how many PRIMARY chunks are kept, out of top_k,
# before remaining slots go to SECONDARY. Kept smaller than top_k
# itself so PRIMARY can't consume the whole budget on its own — with
# several matched documents now sharing this cap, it's a bit more
# generous than the old single-document default. Only granted in full
# to a confident match — see PRIMARY_STRONG_MATCH_THRESHOLD.
PRIMARY_TOP_K_DEFAULT = 4

# Minimum dense cosine score (same scale as min_relevance_score),
# among PRIMARY's own raw candidates before any filtering, for PRIMARY
# to be trusted with its full primary_top_k budget. Below this, a
# topic name can still have matched correctly in name only, without
# its document's actual content being a strong match for this specific
# question — observed in practice: PRIMARY claiming all 4 of its slots
# on a ~0.6-scored wrong document while a different, unscoped document
# scored 0.86+ but only had 1 leftover SECONDARY slot to compete for.
# PRIMARY still gets PRIMARY_WEAK_MATCH_TOP_K slots even below this —
# the topic match isn't discarded, just not trusted with most of the
# budget — and nothing here touches reranker reordering: a confident
# PRIMARY match is still never evicted by a low reranker score (see
# _search's trusted_pool docstring for why that's deliberate).
PRIMARY_STRONG_MATCH_THRESHOLD = 0.65

# How many PRIMARY slots a match below PRIMARY_STRONG_MATCH_THRESHOLD
# still gets, instead of the full primary_top_k.
PRIMARY_WEAK_MATCH_TOP_K = 1

FOCUSSED_CATEGORY_MODE = "FOCUSSED_CATEGORY"
GLOBAL_MODE = "GLOBAL"

# Sentinel for retrieve()'s category_override: an explicit "search
# every category" (the UI's "All categories" option), behaviorally
# identical to not passing category_override at all (None) — kept as
# its own name so dashboard.py's intent reads clearly at the call
# site. Any other string hard-scopes retrieval to that one category —
# see FOCUSSED_CATEGORY_MODE above. There is no third "auto-detect"
# state: a query is always either scoped to a named category or
# fully global.
ALL_CATEGORIES = "__all_categories__"


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    relevance_score: float
    # "dense" (embedding only), "lexical" (BM25 only), or "hybrid"
    # (found by both). PRIMARY chunks are always "dense" or "hybrid".
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


def _group_topics_by_category(doc_topics: list) -> dict:
    by_category = {}
    for topic in doc_topics:
        if topic.category:
            by_category.setdefault(topic.category, []).append(topic)
    return by_category


def _candidate_trace(candidate) -> dict:
    """candidate is a FusedCandidate for every case except PRIMARY's
    reranker_scores (there, trusted_pool scores the already-converted
    RetrievedChunk objects, which carry source_file/page_number
    directly instead of nested under .metadata, and have no separate
    dense/lexical/fused score of their own — cross-reference those
    against this same stage's candidates_considered by source_file/page
    if needed)."""
    if isinstance(candidate, RetrievedChunk):
        return {
            "source_file": candidate.source_file,
            "page_number": candidate.page_number,
            "dense_score": None,
            "lexical_score": None,
            "fused_score": None,
        }
    return {
        "source_file": candidate.metadata.get("source_file"),
        "page_number": candidate.metadata.get("page_number"),
        "dense_score": candidate.dense_score,
        "lexical_score": candidate.lexical_score,
        "fused_score": candidate.fused_score,
    }


def _chunk_trace(chunk: RetrievedChunk) -> dict:
    return {
        "source_file": chunk.source_file,
        "page_number": chunk.page_number,
        "relevance_score": chunk.relevance_score,
        "match_type": chunk.match_type,
        "low_confidence": chunk.low_confidence,
    }


def _stage_trace(label: str, candidates: list, results: list, reranker_scores: list) -> dict:
    """One PRIMARY/SECONDARY/fallback search's full detail, for the
    per-query JSON debug log (see rag._09_common.query_logger):
    every fused candidate considered (with its dense/lexical/fused
    scores), the reranker's own score for every candidate it actually
    scored (empty when no reranker ran this stage), and the chunks
    that survived to become `results`."""
    return {
        "stage": label,
        "candidates_considered": [_candidate_trace(candidate) for candidate in candidates],
        "reranker_scores": [
            {**_candidate_trace(candidate), "reranker_score": score} for candidate, score in reranker_scores
        ],
        "kept_chunks": [_chunk_trace(chunk) for chunk in results],
    }


class RetrievalService:
    """Finds the policy chunks most relevant to a user question.

    Retrieves from topic-matched document(s) first (PRIMARY) — scoped
    to one category when the caller names one, otherwise across the
    whole corpus — and falls back to a plain hybrid (dense + BM25)
    search over that same scope when nothing in PRIMARY is relevant.
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
        primary_top_k: int = PRIMARY_TOP_K_DEFAULT,
        primary_candidates: int = PRIMARY_CANDIDATES,
        primary_strong_match_threshold: float = PRIMARY_STRONG_MATCH_THRESHOLD,
        primary_weak_match_top_k: int = PRIMARY_WEAK_MATCH_TOP_K,
        topic_match_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        topic_match_margin: float = DEFAULT_SIMILARITY_MARGIN,
        topic_match_max_matches: int = DEFAULT_MAX_MATCHES,
        low_confidence_min_score: float = 0.05,
        institution_name: str = None,
    ):
        self._dense_retriever = dense_retriever
        # Public (no leading underscore): main_pipeline.py reuses this
        # same loaded model for similar_question_finder.py instead of
        # loading a second instance of it.
        self.embedding_generator = embedding_generator
        self._top_k = top_k
        self._min_reranker_score = min_reranker_score
        # Only used for PRIMARY search — see _search's trusted_pool
        # argument. Matched document(s) are already trusted by the
        # topic match, so PRIMARY relevance is decided from the dense
        # score, same as before reranker_score became the gate for
        # SECONDARY/GLOBAL. This is on the dense cosine [0, 1] scale,
        # not the reranker's scale.
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
        self._primary_top_k = primary_top_k
        self._primary_candidates = primary_candidates
        self._primary_strong_match_threshold = primary_strong_match_threshold
        self._primary_weak_match_top_k = primary_weak_match_top_k
        self._institution_name = institution_name
        doc_topics = doc_topics or []
        self._global_topic_matcher = EmbeddingTopicMatcher(
            doc_topics,
            embedding_generator,
            similarity_threshold=topic_match_threshold,
            similarity_margin=topic_match_margin,
            max_matches=topic_match_max_matches,
        )
        # One EmbeddingTopicMatcher per category, built once here from
        # doc_topics grouped by their .category — so a FOCUSSED_CATEGORY
        # query only ever gets matched against that category's own
        # documents. Rebuilding RetrievalService (see main_pipeline.py)
        # after any change to categories/documents keeps this current;
        # see admin_page.py's post-rebuild cache invalidation.
        self._category_topic_matchers = {
            category: EmbeddingTopicMatcher(
                topics,
                embedding_generator,
                similarity_threshold=topic_match_threshold,
                similarity_margin=topic_match_margin,
                max_matches=topic_match_max_matches,
            )
            for category, topics in _group_topics_by_category(doc_topics).items()
        }
        # The full structured trace (candidates considered, every
        # reranker score, final kept chunks, per stage) of the most
        # recent retrieve() call — overwritten every call. Read by
        # rag_pipeline.py right after calling retrieve() to build its
        # per-query JSON debug log entry (see rag._09_common.query_logger).
        # None until the first retrieve() call.
        self.last_trace = None

    def match_topics(self, query: str, query_embedding: list = None, category: str = None) -> list:
        """Returns every DocTopic query is about, most similar first —
        empty if none clear the similarity threshold.

        Scoped to `category`'s own documents when given, otherwise
        matched across the whole corpus. A category with no documents
        (or one that no longer exists) simply has no matcher, so this
        returns empty rather than raising.

        Matches against query with the institution's own name stripped
        (see query_expander.strip_institution_name) — it's noise for
        topic-NAME similarity specifically, unlike for the actual
        document search in retrieve(), which keeps it. query_embedding
        (if given, e.g. reused from retrieve()'s dense-search
        embedding) is for the ORIGINAL query, so it's only reused
        as-is when there was nothing to strip; otherwise a fresh
        embedding is computed for the stripped text.
        """
        stripped_query = strip_institution_name(query, self._institution_name)
        if stripped_query != query:
            query_embedding = None
        matcher = self._category_topic_matchers.get(category) if category else self._global_topic_matcher
        if matcher is None:
            return []
        return matcher.match_all(stripped_query, query_embedding=query_embedding)

    def retrieve(
        self,
        query: str,
        metadata_filter: dict = None,
        top_k: int = None,
        min_reranker_score: float = None,
        category_override: str = None,
    ) -> tuple:
        """Returns (chunks, matched_topics) for query, chunks most
        relevant first.

        top_k / min_reranker_score, if given, override this service's
        configured defaults for this call only (used by evaluation
        scripts). matched_topics is returned here (instead of
        requiring a second match_topics() call) so a caller that also
        needs it, like rag_pipeline.py, doesn't re-embed and re-match
        the same query a second time.

        Retrieval always runs on query alone — no conversation history
        is ever considered here.

        category_override (see dashboard.py's Chat page category
        picker) controls scope:
        - None or ALL_CATEGORIES ("all categories" in the UI): GLOBAL
          mode — topic-matching runs across every document.
        - a concrete category name (the user picked one explicitly):
          FOCUSSED_CATEGORY mode — topic-matching runs only against
          that category's documents, and every fallback/SECONDARY
          search stays scoped to it too.

        Flow (same shape in both modes, just different scope):
        1. Topic-match the query within scope. If it maps to one or
           more documents (PRIMARY), search their union for up to
           primary_top_k chunks.
        2. If PRIMARY comes back empty, the topic match isn't trusted
           for this query (or none matched at all) — fall back to a
           plain hybrid search over the same scope instead.
        3. Otherwise, fill whatever's left of top_k with a second,
           same-scope search that excludes the matched document(s)
           (SECONDARY), so PRIMARY + SECONDARY together total top_k.
        """
        top_k = self._top_k if top_k is None else top_k
        min_reranker_score = self._min_reranker_score if min_reranker_score is None else min_reranker_score

        overall_start = time.perf_counter()

        query_embedding = self.embedding_generator.embed_query(query)

        category = category_override if (category_override and category_override != ALL_CATEGORIES) else None
        mode = FOCUSSED_CATEGORY_MODE if category else GLOBAL_MODE
        fallback_filter = {**(metadata_filter or {}), "category": category} if category else metadata_filter

        matched_topics = []
        if metadata_filter is None:
            matched_topics = self.match_topics(query, query_embedding=query_embedding, category=category)

        mapped_documents = sorted({
            filename for topic in matched_topics for filename in topic.docs_filename
        })

        primary_results = None
        secondary_results = None
        stage_traces = []

        if mapped_documents:
            scoped_filter = {"source_file": {"$in": mapped_documents}}
            primary_keep = min(self._primary_top_k, top_k)
            primary_candidates, primary_results, primary_reranker_scores = self._search(
                query, query_embedding, scoped_filter, self._primary_candidates,
                keep=primary_keep, min_reranker_score=min_reranker_score, trusted_pool=True,
            )
            stage_traces.append(_stage_trace("primary", primary_candidates, primary_results, primary_reranker_scores))

        # Falls through here both when nothing matched at all, and
        # when a topic matched but PRIMARY turned up nothing usable
        # (primary_results empty) — a possibly-wrong topic match isn't
        # trusted over a plain search of the same scope.
        if not primary_results:
            primary_results = None
            candidates, results, reranker_scores = self._search(
                query, query_embedding, fallback_filter, self._hybrid_candidates,
                keep=top_k, min_reranker_score=min_reranker_score,
            )
            stage_traces.append(_stage_trace("fallback", candidates, results, reranker_scores))
        else:
            secondary_keep = top_k - len(primary_results)
            secondary_filter = {"category": category} if category else None
            if secondary_keep > 0:
                secondary_candidates, secondary_results, secondary_reranker_scores = self._search(
                    query, query_embedding, secondary_filter, self._hybrid_candidates,
                    keep=secondary_keep, min_reranker_score=min_reranker_score,
                    exclude_source_files=mapped_documents,
                )
            else:
                secondary_candidates, secondary_results, secondary_reranker_scores = [], [], []
            stage_traces.append(_stage_trace("secondary", secondary_candidates, secondary_results, secondary_reranker_scores))
            candidates = primary_candidates + secondary_candidates
            results = primary_results + secondary_results

        total_seconds = time.perf_counter() - overall_start
        self._log_summary(
            query, matched_topics, mode, candidates, results, total_seconds,
            primary_results=primary_results, secondary_results=secondary_results, category=category,
        )
        self.last_trace = {
            "query": query,
            "mode": mode,
            "category": category,
            "matched_topics": [
                {"topic": topic.topic, "docs_filename": topic.docs_filename} for topic in matched_topics
            ],
            "stages": stage_traces,
            "retrieval_seconds": total_seconds,
        }
        return results, matched_topics

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

        trusted_pool is True only for the PRIMARY call: that pool is
        already trusted by the topic match, so relevance is decided by
        dense score, and the reranker (if any) only reorders that
        already-relevant pool — it can reorder PRIMARY results but
        can't drop one outright. This matters because a cross-encoder
        can genuinely score a correct chunk low (e.g. a short heading
        like "Welfare Measures for Staff"), and letting that evict a
        matched document's own top chunk just forces a worse fallback.
        SECONDARY/GLOBAL pools are not from a trusted document, so
        reranker_score stays the hard gate there.

        `keep` itself is shrunk to primary_weak_match_top_k (before any
        of the above) when trusted_pool's own best dense score among
        its raw candidates doesn't clear primary_strong_match_threshold
        — a topic NAME match doesn't guarantee the document's CONTENT
        is actually a strong match for this specific question, and a
        weak one shouldn't be trusted with most of top_k's budget at
        the expense of what SECONDARY might find. This only changes how
        much of the budget PRIMARY gets, never which of its own
        candidates the reranker is allowed to keep — the no-eviction
        guarantee above is unaffected.
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
            return fused_candidates, [], []

        if trusted_pool:
            best_dense_score = max(
                (candidate.dense_score for candidate in fused_candidates if candidate.dense_score is not None),
                default=0.0,
            )
            if best_dense_score < self._primary_strong_match_threshold:
                keep = min(keep, self._primary_weak_match_top_k)
            relevant_chunks = self._filter_relevant_by_dense_score(fused_candidates, self._min_relevance_score)
            if self._reranker and relevant_chunks:
                # Scored (not .rerank()'d) directly so the per-candidate
                # scores survive for the query log, same shape as the
                # non-trusted path below — .rerank() alone would reorder
                # and truncate without ever handing the scores back.
                scores = self._reranker.score(query, relevant_chunks)
                reranker_scores = list(zip(relevant_chunks, (float(score) for score in scores)))
                reranker_scores.sort(key=operator.itemgetter(1), reverse=True)
                ordered_chunks = [chunk for chunk, _ in reranker_scores]
                return fused_candidates, ordered_chunks[:keep], reranker_scores
            return fused_candidates, relevant_chunks[:keep], []

        relevant_chunks, reranker_scores = self._filter_relevant(query, fused_candidates, min_reranker_score)
        return fused_candidates, relevant_chunks[:keep], reranker_scores

    def _filter_relevant(self, query: str, fused_candidates: list, min_reranker_score: float) -> tuple:
        """Turns fused candidates into (RetrievedChunk objects clearing
        min_reranker_score, every (candidate, score) pair the reranker
        actually scored — [] when no reranker is configured), most
        relevant first. This is the full relevant pool — callers cap
        it to `keep` themselves, after this filtering has already run.

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
        return self._filter_relevant_by_dense_score(fused_candidates, min_reranker_score), []

    def _filter_relevant_by_reranker(self, query: str, fused_candidates: list, min_reranker_score: float) -> tuple:
        """Returns (chunks, scored_pairs) — see _filter_relevant."""
        pool = fused_candidates[: self._rerank_candidates]
        scores = self._reranker.score(query, pool)
        scored_pairs = [(candidate, float(score)) for candidate, score in zip(pool, scores)]
        scored_pairs.sort(key=operator.itemgetter(1), reverse=True)

        chunks = []
        for candidate, score in scored_pairs:
            if score < min_reranker_score:
                continue
            chunks.append(_chunk_from_candidate(candidate, score))

        if chunks:
            return chunks, scored_pairs

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
            return [], scored_pairs
        return [_chunk_from_candidate(best_candidate, best_score, low_confidence=True)], scored_pairs

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
        self, query, matched_topics, mode, candidates, results, total_seconds,
        primary_results=None, secondary_results=None, category=None,
    ) -> None:
        """Always-on, one-line-per-query summary: query, detected
        topic(s), matched document(s), retrieval mode, candidate/result
        counts, and final source documents."""
        sources = sorted({chunk.source_file for chunk in results})
        mapped_documents = sorted({filename for topic in matched_topics for filename in topic.docs_filename})
        logger.info(
            "query=%r topics=%r mapped_documents=%r category=%s mode=%s candidates=%d "
            "final_chunks=%d sources=%s seconds=%.3f",
            query,
            [topic.topic for topic in matched_topics],
            mapped_documents,
            category,
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
