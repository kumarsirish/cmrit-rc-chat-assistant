import operator
from dataclasses import dataclass, field

from rag._04_retrieval.bm25_retriever import BM25Document, BM25Index
from rag._04_retrieval.retrieval_service import GLOBAL_MODE, MAPPED_DOCUMENT_MODE, RetrievalService, RetrievedChunk
from rag._03_routing.docs_metadata import DocTopic


def _matches(actual, expected):
    """Same {"$in": [...]} membership support as the real Chroma/BM25
    filters, for FakeDenseRetriever's exact-match filtering below."""
    if isinstance(expected, dict) and "$in" in expected:
        return actual in expected["$in"]
    return actual == expected


class FakeEmbeddingGenerator:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_texts(self, texts):
        return [self.embed_query(text) for text in texts]


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeDenseRetriever:
    """Same distances/metadatas -> (Document, score) shape as the real
    DenseRetriever, but with results fixed up front instead of coming
    from Chroma."""

    def __init__(self, distances, metadatas=None):
        self._distances = distances
        self._metadatas = metadatas
        self.search_call_count = 0
        self.last_filter = None

    def search(self, query_embedding, top_k, filter=None):
        self.search_call_count += 1
        self.last_filter = filter
        metadatas = self._metadatas or [
            {"source_file": f"doc{i}.pdf", "page_number": i + 1, "chunk_index": 0}
            for i in range(len(self._distances))
        ]
        results = [
            (FakeDocument(page_content=f"chunk text {i}", metadata=metadatas[i]), 1.0 - distance)
            for i, distance in enumerate(self._distances)
        ]
        if filter:
            results = [
                (document, score)
                for document, score in results
                if all(_matches(document.metadata.get(key), value) for key, value in filter.items())
            ]
        return results[:top_k]


class FakeReranker:
    """Returns a fixed score per chunk text, looked up from a dict, so
    tests can control exactly which candidates pass/fail the relevance
    threshold and in what order."""

    def __init__(self, scores_by_text):
        self._scores_by_text = scores_by_text

    def score(self, query, chunks):
        return [self._scores_by_text[chunk.text] for chunk in chunks]

    def rerank(self, query, chunks, top_k):
        if not chunks:
            return chunks
        scores = self.score(query, chunks)
        scored_pairs = list(zip(scores, chunks))
        scored_pairs.sort(key=operator.itemgetter(0), reverse=True)
        return [chunk for _, chunk in scored_pairs[:top_k]]


def _make_bm25_index(entries):
    """entries: list of (text, source_file, page_number)."""
    documents = [
        BM25Document(text=text, metadata={"source_file": source_file, "page_number": page_number, "chunk_index": 0})
        for text, source_file, page_number in entries
    ]
    return BM25Index(documents)


# --- dense-only relevance, no reranker (fallback behavior) ---------------


def test_retrieve_filters_out_low_relevance_dense_only_chunks():
    dense_retriever = FakeDenseRetriever(distances=[0.1, 0.9])
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.35,
    )
    results, _ = service.retrieve("some question")
    assert len(results) == 1
    assert results[0].source_file == "doc0.pdf"
    assert results[0].relevance_score == 0.9
    assert results[0].match_type == "dense"


def test_retrieve_returns_empty_list_when_nothing_is_relevant():
    dense_retriever = FakeDenseRetriever(distances=[0.8, 0.95])
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.35,
    )
    results, _ = service.retrieve("unrelated question")
    assert results == []


def test_retrieve_surfaces_lexical_only_match_below_dense_relevance_threshold():
    dense_retriever = FakeDenseRetriever(distances=[0.9])
    bm25_index = _make_bm25_index([("Leave application through Darwinbox portal", "leave.pdf", 2)])
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_reranker_score=0.35,
        bm25_index=bm25_index,
    )
    results, _ = service.retrieve("what is darwinbox")
    assert len(results) == 1
    assert results[0].source_file == "leave.pdf"
    assert results[0].match_type == "lexical"


def test_retrieve_marks_chunk_found_by_both_searches_as_hybrid():
    dense_retriever = FakeDenseRetriever(
        distances=[0.1], metadatas=[{"source_file": "leave.pdf", "page_number": 2, "chunk_index": 0}]
    )
    bm25_index = _make_bm25_index([("Leave application through Darwinbox portal", "leave.pdf", 2)])
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_reranker_score=0.35,
        bm25_index=bm25_index,
    )
    results, _ = service.retrieve("darwinbox leave")
    assert len(results) == 1
    assert results[0].match_type == "hybrid"


def test_retrieve_applies_metadata_filter_to_dense_search():
    dense_retriever = FakeDenseRetriever(
        distances=[0.1, 0.1],
        metadatas=[
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "probation.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_reranker_score=0.35,
    )
    results, _ = service.retrieve("policy question", metadata_filter={"source_file": "leave.pdf"})
    assert len(results) == 1
    assert results[0].source_file == "leave.pdf"


# --- reranker-driven relevance -------------------------------------------


def test_bm25_only_candidate_is_subject_to_the_reranker_threshold():
    # With a reranker configured, the old "BM25-only skips the
    # threshold" exemption no longer applies: a lexical-only hit is
    # judged by min_reranker_score just like any other candidate. It
    # still comes back as the low-confidence fallback (see
    # test_low_confidence_fallback_* below) since 0.2 clears the
    # default low_confidence_min_score of 0.05.
    dense_retriever = FakeDenseRetriever(distances=[0.9])  # dense finds nothing relevant
    bm25_index = _make_bm25_index([("Leave application through Darwinbox portal", "leave.pdf", 2)])
    reranker = FakeReranker({"chunk text 0": 0.1, "Leave application through Darwinbox portal": 0.2})
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_reranker_score=0.35,
        bm25_index=bm25_index,
        reranker=reranker,
        rerank_candidates=8,
    )
    results, _ = service.retrieve("what is darwinbox")
    # The lexical hit scored 0.2, below min_reranker_score=0.35, so it
    # doesn't count as a normal match — but it's still the best
    # candidate available and clears low_confidence_min_score, so it
    # comes back flagged instead of being dropped outright.
    assert len(results) == 1
    assert results[0].low_confidence is True
    assert results[0].relevance_score == 0.2


def test_low_confidence_fallback_returns_best_candidate_below_threshold():
    dense_retriever = FakeDenseRetriever(distances=[0.9, 0.9])  # dense finds nothing relevant
    reranker = FakeReranker({"chunk text 0": 0.10, "chunk text 1": 0.20})
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_reranker_score=0.35,
        reranker=reranker,
        rerank_candidates=8,
        low_confidence_min_score=0.05,
    )
    results, _ = service.retrieve("some question")
    # Neither candidate clears 0.35, so normally this would refuse.
    # Instead, the single best-scoring one (0.20) is returned, flagged
    # low_confidence, since it clears low_confidence_min_score.
    assert len(results) == 1
    assert results[0].source_file == "doc1.pdf"
    assert results[0].relevance_score == 0.20
    assert results[0].low_confidence is True


def test_low_confidence_fallback_still_refuses_below_its_own_floor():
    dense_retriever = FakeDenseRetriever(distances=[0.9])  # dense finds nothing relevant
    reranker = FakeReranker({"chunk text 0": 0.02})
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_reranker_score=0.35,
        reranker=reranker,
        rerank_candidates=8,
        low_confidence_min_score=0.05,
    )
    results, _ = service.retrieve("totally unrelated question")
    # 0.02 is below even low_confidence_min_score, so this is treated
    # as genuinely unanswerable — no fallback chunk is returned.
    assert results == []


def test_primary_cap_applies_after_dense_relevance_filter():
    # PRIMARY is a trusted pool (see below): its relevance filter is
    # the dense score, not the reranker. Dense scores: A=0.91
    # (relevant), B=0.35 (below threshold), C=0.84 (relevant), D=0.20
    # (below threshold). primary_top_k=2 must keep [A, C] (filter
    # first, then cap) — not cap to the first 2 candidates and filter
    # after, which would lose C.
    class LabeledDenseRetriever:
        def search(self, query_embedding, top_k, filter=None):
            docs = [
                (FakeDocument(page_content="B", metadata={"source_file": "mapped.pdf", "page_number": 1, "chunk_index": 0}), 0.35),
                (FakeDocument(page_content="D", metadata={"source_file": "mapped.pdf", "page_number": 2, "chunk_index": 0}), 0.20),
                (FakeDocument(page_content="A", metadata={"source_file": "mapped.pdf", "page_number": 3, "chunk_index": 0}), 0.91),
                (FakeDocument(page_content="C", metadata={"source_file": "mapped.pdf", "page_number": 4, "chunk_index": 0}), 0.84),
            ]
            return docs[:top_k]

    doc_topics = [DocTopic(topic="Mapped Policy", link="https://example.com/mapped", docs_filename="mapped.pdf")]
    service = RetrievalService(
        dense_retriever=LabeledDenseRetriever(),
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.5,
        min_relevance_score=0.5,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=2,
    )
    results, _ = service.retrieve("what is the mapped policy")
    assert [chunk.text for chunk in results] == ["A", "C"]


def test_primary_relevance_is_decided_by_dense_score_not_reranker_score():
    # Regression case: a cross-encoder can genuinely score a correct
    # chunk low (observed for real: a reranker scored the correct
    # "Welfare Measures for Staff" heading chunk 0.05-ish, well below
    # any sane threshold, even though it was the #1 dense AND lexical
    # hit in its own document). The mapped document is already
    # trusted by the topic match, so PRIMARY must not drop this chunk
    # just because the reranker underscores it — the reranker may
    # still reorder within PRIMARY, but not evict.
    class LabeledDenseRetriever:
        def search(self, query_embedding, top_k, filter=None):
            return [
                (FakeDocument(page_content="A", metadata={"source_file": "mapped.pdf", "page_number": 1, "chunk_index": 0}), 0.60),
            ][:top_k]

    reranker = FakeReranker({"A": 0.05})
    doc_topics = [DocTopic(topic="Mapped Policy", link="https://example.com/mapped", docs_filename="mapped.pdf")]
    service = RetrievalService(
        dense_retriever=LabeledDenseRetriever(),
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.5,
        min_relevance_score=0.40,
        reranker=reranker,
        rerank_candidates=8,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=2,
    )
    results, _ = service.retrieve("what is the mapped policy")
    assert [chunk.text for chunk in results] == ["A"]


def test_primary_reranker_reorders_but_does_not_evict_dense_relevant_chunks():
    class LabeledDenseRetriever:
        def search(self, query_embedding, top_k, filter=None):
            return [
                (FakeDocument(page_content="A", metadata={"source_file": "mapped.pdf", "page_number": 1, "chunk_index": 0}), 0.60),
                (FakeDocument(page_content="B", metadata={"source_file": "mapped.pdf", "page_number": 2, "chunk_index": 0}), 0.55),
            ][:top_k]

    # Both A and B pass the dense-score gate; the reranker prefers B.
    reranker = FakeReranker({"A": 0.10, "B": 0.90})
    doc_topics = [DocTopic(topic="Mapped Policy", link="https://example.com/mapped", docs_filename="mapped.pdf")]
    service = RetrievalService(
        dense_retriever=LabeledDenseRetriever(),
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.5,
        min_relevance_score=0.40,
        reranker=reranker,
        rerank_candidates=8,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=2,
    )
    results, _ = service.retrieve("what is the mapped policy")
    assert [chunk.text for chunk in results] == ["B", "A"]


def test_secondary_slot_fill_applies_after_relevance_filter():
    # Same filter-then-cap bug, applied to SECONDARY's remaining_slots
    # truncation: PRIMARY fills one slot, SECONDARY has one remaining
    # slot and candidates E=0.9 (relevant), F=0.6 (relevant), scored so
    # the stronger one (E) must be kept, not whichever appeared first
    # in the fused/RRF order.
    class LabeledDenseRetriever:
        def __init__(self):
            self.calls = []

        def search(self, query_embedding, top_k, filter=None):
            self.calls.append(filter)
            if filter:  # PRIMARY call, scoped to mapped.pdf
                return [
                    (FakeDocument(page_content="P1", metadata={"source_file": "mapped.pdf", "page_number": 1, "chunk_index": 0}), 0.9),
                ]
            # SECONDARY call, unscoped
            return [
                (FakeDocument(page_content="F", metadata={"source_file": "other.pdf", "page_number": 1, "chunk_index": 0}), 0.6),
                (FakeDocument(page_content="E", metadata={"source_file": "other.pdf", "page_number": 2, "chunk_index": 0}), 0.9),
            ]

    reranker = FakeReranker({"P1": 0.9, "F": 0.3, "E": 0.8})
    doc_topics = [DocTopic(topic="Mapped Policy", link="https://example.com/mapped", docs_filename="mapped.pdf")]
    service = RetrievalService(
        dense_retriever=LabeledDenseRetriever(),
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.5,
        reranker=reranker,
        rerank_candidates=8,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=2,
    )
    results, _ = service.retrieve("what is the mapped policy")
    # PRIMARY = [P1]. SECONDARY has 1 slot; F fails the threshold
    # (0.3 < 0.5), only E qualifies, so E must be the one kept.
    assert [chunk.text for chunk in results] == ["P1", "E"]


# --- PRIMARY / SECONDARY / GLOBAL routing ---------------------------------


def test_topic_match_puts_mapped_document_chunks_first_but_allows_secondary_fill():
    dense_retriever = FakeDenseRetriever(
        distances=[0.02, 0.6],
        metadatas=[
            {"source_file": "doc0.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
    )
    results, _ = service.retrieve("what is the leave policy")
    assert [chunk.source_file for chunk in results] == ["leave.pdf", "doc0.pdf"]
    assert dense_retriever.last_filter is None  # last call is the SECONDARY (unfiltered) search


def test_primary_full_skips_secondary_search_entirely():
    # top_k=1 leaves PRIMARY no room to share with SECONDARY.
    dense_retriever = FakeDenseRetriever(
        distances=[0.02, 0.6],
        metadatas=[
            {"source_file": "doc0.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=1,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
    )
    results, _ = service.retrieve("what is the leave policy")
    assert [chunk.source_file for chunk in results] == ["leave.pdf"]
    assert dense_retriever.search_call_count == 1
    assert dense_retriever.last_filter == {"source_file": {"$in": ["leave.pdf"]}}


def test_topic_mapped_to_multiple_documents_searches_all_of_them_as_primary():
    # A topic can be backed by more than one PDF (docs_filename with
    # several entries); PRIMARY must pull matching chunks from any of
    # them, and SECONDARY must exclude all of them, not just the first.
    dense_retriever = FakeDenseRetriever(
        distances=[0.02, 0.05, 0.6],
        metadatas=[
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "leave-addendum.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "doc0.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [
        DocTopic(
            topic="Leave Policy",
            link="https://example.com/leave",
            docs_filename=["leave.pdf", "leave-addendum.pdf"],
        )
    ]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=3,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=2,
    )
    results, _ = service.retrieve("what is the leave policy")
    assert [chunk.source_file for chunk in results] == ["leave.pdf", "leave-addendum.pdf", "doc0.pdf"]
    assert dense_retriever.last_filter is None  # last call is the SECONDARY (unfiltered) search


def test_primary_full_with_overridden_config_still_skips_secondary():
    # Under shipped config (primary_top_k < top_k) FULL never happens.
    # Override both to the same value to actually exercise it.
    dense_retriever = FakeDenseRetriever(
        distances=[0.1, 0.1, 0.1],
        metadatas=[
            {"source_file": "mapped.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "mapped.pdf", "page_number": 2, "chunk_index": 0},
            {"source_file": "mapped.pdf", "page_number": 3, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Mapped Policy", link="https://example.com/mapped", docs_filename="mapped.pdf")]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=3,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=3,
    )
    results, _ = service.retrieve("what is the mapped policy")
    assert len(results) == 3
    assert all(chunk.source_file == "mapped.pdf" for chunk in results)
    assert dense_retriever.search_call_count == 1  # PRIMARY only, no SECONDARY call


def test_secondary_excludes_the_entire_mapped_document_not_just_returned_chunks():
    # mapped.pdf has 3 relevant chunks but primary_top_k only lets 1
    # into PRIMARY. None of the other 2 (never selected by PRIMARY)
    # may leak into SECONDARY's results.
    class LabeledDenseRetriever:
        def search(self, query_embedding, top_k, filter=None):
            if filter:  # PRIMARY, scoped
                return [
                    (FakeDocument(page_content="M1", metadata={"source_file": "mapped.pdf", "page_number": 1, "chunk_index": 0}), 0.95),
                ][:top_k]
            # SECONDARY, unscoped — includes leftover mapped.pdf chunks
            # a correct implementation must still filter out.
            return [
                (FakeDocument(page_content="M2", metadata={"source_file": "mapped.pdf", "page_number": 2, "chunk_index": 0}), 0.9),
                (FakeDocument(page_content="M3", metadata={"source_file": "mapped.pdf", "page_number": 3, "chunk_index": 0}), 0.85),
                (FakeDocument(page_content="O1", metadata={"source_file": "other.pdf", "page_number": 1, "chunk_index": 0}), 0.5),
            ][:top_k]

    doc_topics = [DocTopic(topic="Mapped Policy", link="https://example.com/mapped", docs_filename="mapped.pdf")]
    service = RetrievalService(
        dense_retriever=LabeledDenseRetriever(),
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=3,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=1,
    )
    results, _ = service.retrieve("what is the mapped policy")
    sources = [chunk.source_file for chunk in results]
    assert sources.count("mapped.pdf") == 1  # only the PRIMARY chunk
    assert "other.pdf" in sources


def test_topic_match_with_no_local_document_falls_back_to_global_mode():
    dense_retriever = FakeDenseRetriever(distances=[0.1])
    doc_topics = [DocTopic(topic="External Portal", link="https://example.com/portal", docs_filename=None)]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
    )
    results, _ = service.retrieve("what is the external portal")
    assert len(results) == 1
    assert dense_retriever.last_filter is None


def test_explicit_metadata_filter_overrides_topic_detection():
    dense_retriever = FakeDenseRetriever(
        distances=[0.1, 0.1],
        metadatas=[
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "probation.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
    )
    results, _ = service.retrieve("what is the leave policy", metadata_filter={"source_file": "probation.pdf"})
    assert [chunk.source_file for chunk in results] == ["probation.pdf"]


def test_retrieve_returns_matched_topic_alongside_chunks():
    # This is what lets rag_pipeline.py avoid a second match_topic()
    # call (and a second query embedding) for the same question.
    dense_retriever = FakeDenseRetriever(
        distances=[0.1], metadatas=[{"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0}]
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
    )
    _, matched_topic = service.retrieve("what is the leave policy")
    assert matched_topic.topic == "Leave Policy"


def test_retrieve_logs_mode_and_mapped_document(caplog):
    import logging

    dense_retriever = FakeDenseRetriever(
        distances=[0.1], metadatas=[{"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0}]
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.35,
        doc_topics=doc_topics,
    )
    with caplog.at_level(logging.INFO, logger="rag._04_retrieval.retrieval_service"):
        service.retrieve("what is the leave policy")
    assert any(
        f"mode={MAPPED_DOCUMENT_MODE}" in record.message and "leave.pdf" in record.message
        for record in caplog.records
    )


def test_retrieve_logs_global_mode_when_no_topic_matches(caplog):
    import logging

    dense_retriever = FakeDenseRetriever(distances=[0.9])
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.35,
    )
    with caplog.at_level(logging.INFO, logger="rag._04_retrieval.retrieval_service"):
        service.retrieve("totally unrelated question")
    assert any(f"mode={GLOBAL_MODE}" in record.message for record in caplog.records)


# --- per-call overrides (used by the UI's retrieval sliders) -------------


def test_retrieve_top_k_override_replaces_the_configured_default():
    dense_retriever = FakeDenseRetriever(distances=[0.1, 0.1, 0.1])
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=1,
        min_reranker_score=0.35,
    )
    results, _ = service.retrieve("some question", top_k=3)
    assert len(results) == 3


def test_retrieve_min_reranker_score_override_replaces_the_configured_default():
    dense_retriever = FakeDenseRetriever(distances=[0.5])  # relevance 0.5
    service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_reranker_score=0.9,  # would normally drop a 0.5 match
    )
    results, _ = service.retrieve("some question", min_reranker_score=0.3)
    assert len(results) == 1


# --- match_topic institution-name stripping --------------------------------


class TextSensitiveEmbeddingGenerator:
    """Unlike FakeEmbeddingGenerator, returns a different vector
    depending on whether "CMRIT" is still in the text, so these tests
    can tell whether match_topic() matched against the original query
    or the institution-name-stripped one."""

    def __init__(self):
        self.embed_query_call_count = 0

    def embed_query(self, text):
        self.embed_query_call_count += 1
        return [0.0, 1.0, 0.0] if "CMRIT" in text else [1.0, 0.0, 0.0]

    def embed_texts(self, texts):
        return [self.embed_query(text) for text in texts]


def test_match_topic_strips_institution_name_before_matching():
    embedding_generator = TextSensitiveEmbeddingGenerator()
    topic = DocTopic(topic="Some Topic", link="https://example.com/topic", docs_filename=["some.pdf"])
    service = RetrievalService(
        dense_retriever=FakeDenseRetriever(distances=[]),
        embedding_generator=embedding_generator,
        top_k=1,
        min_reranker_score=0.35,
        doc_topics=[topic],
    )
    # The query's own embedding (computed with "CMRIT" still in it)
    # would score 0 against the topic phrase's [1, 0, 0] embedding —
    # matching only works because match_topic re-embeds the stripped text.
    original_embedding = embedding_generator.embed_query("about Some Topic at CMRIT")
    matched = service.match_topic("about Some Topic at CMRIT", query_embedding=original_embedding)
    assert matched is topic


def test_match_topic_reuses_given_embedding_when_nothing_to_strip():
    embedding_generator = TextSensitiveEmbeddingGenerator()
    topic = DocTopic(topic="Some Topic", link="https://example.com/topic", docs_filename=["some.pdf"])
    service = RetrievalService(
        dense_retriever=FakeDenseRetriever(distances=[]),
        embedding_generator=embedding_generator,
        top_k=1,
        min_reranker_score=0.35,
        doc_topics=[topic],
    )
    embedding_generator.embed_query_call_count = 0
    given_embedding = [1.0, 0.0, 0.0]
    matched = service.match_topic("about Some Topic", query_embedding=given_embedding)
    assert matched is topic
    # No "CMRIT" to strip, so the given embedding must be reused as-is
    # rather than paying for a second embed_query call.
    assert embedding_generator.embed_query_call_count == 0
