from dataclasses import dataclass, field

from rag.indexing.bm25_index import BM25Document, BM25Index
from rag.indexing.docs_metadata import DocTopic
from rag.retrieval.retriever import GLOBAL_MODE, MAPPED_DOCUMENT_MODE, RetrievedChunk, Retriever


class FakeEmbeddingGenerator:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeVectorStore:
    def __init__(self, distances, metadatas=None):
        self._distances = distances
        self._metadatas = metadatas
        self.search_call_count = 0
        self.last_filter = None

    def similarity_search_with_relevance_scores(self, query_embedding, top_k, filter=None):
        self.search_call_count += 1
        self.last_filter = filter
        metadatas = self._metadatas or [
            {"source_file": f"doc{i}.pdf", "page_number": i + 1, "chunk_index": 0}
            for i in range(len(self._distances))
        ]
        results = [
            (
                FakeDocument(page_content=f"chunk text {i}", metadata=metadatas[i]),
                1.0 - distance,
            )
            for i, distance in enumerate(self._distances)
        ]
        if filter:
            results = [
                (document, score)
                for document, score in results
                if all(document.metadata.get(key) == value for key, value in filter.items())
            ]
        return results[:top_k]


def _make_bm25_index(entries):
    """entries: list of (text, source_file, page_number)."""
    documents = [
        BM25Document(
            text=text,
            metadata={"source_file": source_file, "page_number": page_number, "chunk_index": 0},
        )
        for text, source_file, page_number in entries
    ]
    return BM25Index(documents)


def test_retrieve_filters_out_low_relevance_dense_only_chunks():
    # distance 0.1 -> relevance 0.9 (kept), distance 0.9 -> relevance 0.1 (dropped)
    vector_store = FakeVectorStore(distances=[0.1, 0.9])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
    )
    results = retriever.retrieve("some question")
    assert len(results) == 1
    assert results[0].source_file == "doc0.pdf"
    assert results[0].relevance_score == 0.9
    assert results[0].match_type == "dense"


def test_retrieve_returns_empty_list_when_nothing_is_relevant():
    vector_store = FakeVectorStore(distances=[0.8, 0.95])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
    )
    results = retriever.retrieve("unrelated question")
    assert results == []


def test_retrieve_surfaces_lexical_only_match_below_dense_relevance_threshold():
    # Dense search alone finds nothing above threshold ...
    vector_store = FakeVectorStore(distances=[0.9])
    # ... but BM25 has a literal token match in a different document.
    bm25_index = _make_bm25_index(
        [("Leave application through Darwinbox portal", "leave.pdf", 2)]
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_relevance_score=0.35,
        bm25_index=bm25_index,
    )
    results = retriever.retrieve("what is darwinbox")
    assert len(results) == 1
    assert results[0].source_file == "leave.pdf"
    assert results[0].match_type == "lexical"


def test_retrieve_marks_chunk_found_by_both_searches_as_hybrid():
    vector_store = FakeVectorStore(
        distances=[0.1],
        metadatas=[{"source_file": "leave.pdf", "page_number": 2, "chunk_index": 0}],
    )
    bm25_index = _make_bm25_index([("Leave application through Darwinbox portal", "leave.pdf", 2)])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_relevance_score=0.35,
        bm25_index=bm25_index,
    )
    results = retriever.retrieve("darwinbox leave")
    assert len(results) == 1
    assert results[0].match_type == "hybrid"


def test_retrieve_returns_empty_when_no_bm25_index_and_dense_finds_nothing():
    vector_store = FakeVectorStore(distances=[0.9])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
    )
    results = retriever.retrieve("what is it")
    assert results == []


def test_retrieve_applies_metadata_filter_to_dense_search():
    vector_store = FakeVectorStore(
        distances=[0.1, 0.1],
        metadatas=[
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "probation.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_relevance_score=0.35,
    )
    results = retriever.retrieve("policy question", metadata_filter={"source_file": "leave.pdf"})
    assert len(results) == 1
    assert results[0].source_file == "leave.pdf"


def test_lexical_only_match_gets_conservative_relevance_score_not_a_perfect_one():
    vector_store = FakeVectorStore(distances=[0.9])
    bm25_index = _make_bm25_index([("Leave application through Darwinbox portal", "leave.pdf", 2)])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_relevance_score=0.35,
        bm25_index=bm25_index,
    )
    results = retriever.retrieve("what is darwinbox")
    assert len(results) == 1
    assert results[0].match_type == "lexical"
    # No longer a fabricated 1.0 confidence sentinel.
    assert results[0].relevance_score == 0.35


# --- MAPPED_DOCUMENT mode: a matched topic with a local document ----------


def test_topic_match_puts_mapped_document_chunks_first_but_allows_a_second_document_to_fill_leftover_slots():
    # doc0.pdf is a much stronger dense match, but it isn't the topic's
    # mapped document — leave.pdf is. Since PRIMARY/MAPPED_DOCUMENT
    # search only turns up one chunk here, and top_k leaves a slot
    # unfilled, a SECONDARY search is allowed to fill it — but leave.pdf
    # (PRIMARY) must still come first regardless of doc0.pdf's much
    # stronger raw score (see module docstring: PRIMARY always
    # precedes SECONDARY).
    vector_store = FakeVectorStore(
        distances=[0.02, 0.6],
        metadatas=[
            {"source_file": "doc0.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
        doc_topics=doc_topics,
    )
    results = retriever.retrieve("what is the leave policy")
    assert [chunk.source_file for chunk in results] == ["leave.pdf", "doc0.pdf"]
    # First (PRIMARY) search is restricted via a single filtered dense
    # search, not a global one.
    assert vector_store.last_filter is None  # last call is the SECONDARY (unfiltered) search


def test_topic_match_restricts_to_mapped_document_only_when_primary_already_fills_top_k():
    # Same setup as above, but top_k=1 leaves PRIMARY no room to share
    # with a SECONDARY search — the single mapped-document chunk must
    # be the entire result, and only one search call should happen,
    # preserving the original single-document guarantee for the case
    # where there's genuinely nothing left to fill.
    vector_store = FakeVectorStore(
        distances=[0.02, 0.6],
        metadatas=[
            {"source_file": "doc0.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=1,
        min_relevance_score=0.35,
        doc_topics=doc_topics,
    )
    results = retriever.retrieve("what is the leave policy")
    assert [chunk.source_file for chunk in results] == ["leave.pdf"]
    assert vector_store.search_call_count == 1
    assert vector_store.last_filter == {"source_file": "leave.pdf"}


def test_topic_match_secondary_search_excludes_the_mapped_document():
    # doc0.pdf is the mapped document; two OTHER documents exist.
    # top_k=3 with PRIMARY finding 1 chunk leaves 2 SECONDARY slots —
    # the SECONDARY search must never bring back doc0.pdf again
    # (that would duplicate the PRIMARY chunk).
    vector_store = FakeVectorStore(
        distances=[0.1, 0.2, 0.3],
        metadatas=[
            {"source_file": "doc0.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "doc1.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "doc2.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Doc0 Policy", link="https://example.com/doc0", docs_filename="doc0.pdf")]
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=3,
        min_relevance_score=0.35,
        doc_topics=doc_topics,
    )
    results = retriever.retrieve("what is doc0 about")
    sources = [chunk.source_file for chunk in results]
    assert sources[0] == "doc0.pdf"
    assert sources.count("doc0.pdf") == 1
    assert set(sources[1:]) == {"doc1.pdf", "doc2.pdf"}


def test_topic_match_with_no_local_document_falls_back_to_global_mode():
    # The topic matches, but has no docs_filename (link-only entry) —
    # nothing to restrict retrieval to, so it must fall back to a
    # normal global search rather than returning nothing.
    vector_store = FakeVectorStore(distances=[0.1])
    doc_topics = [DocTopic(topic="External Portal", link="https://example.com/portal", docs_filename=None)]
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
        doc_topics=doc_topics,
    )
    results = retriever.retrieve("what is the external portal")
    assert len(results) == 1
    assert vector_store.last_filter is None


def test_explicit_metadata_filter_overrides_topic_detection():
    # An explicit metadata_filter from the caller takes priority over
    # topic-based restriction and runs global (filtered) hybrid search.
    vector_store = FakeVectorStore(
        distances=[0.1, 0.1],
        metadatas=[
            {"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "probation.pdf", "page_number": 1, "chunk_index": 0},
        ],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_relevance_score=0.35,
        doc_topics=doc_topics,
    )
    results = retriever.retrieve(
        "what is the leave policy", metadata_filter={"source_file": "probation.pdf"}
    )
    assert [chunk.source_file for chunk in results] == ["probation.pdf"]


def test_reranker_cannot_replace_the_mapped_document_with_another():
    # A reranker that would always prefer some other document must not
    # be able to — in MAPPED_DOCUMENT mode it never even sees chunks
    # from any document but the mapped one.
    vector_store = FakeVectorStore(
        distances=[0.1],
        metadatas=[{"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0}],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]

    class FakeRerankerPreferringOtherDocs:
        def rerank(self, query, chunks, top_k):
            # Even a "hostile" reranker can only reorder/truncate what
            # it's given — it can't conjure a chunk from another doc.
            return list(reversed(chunks))[:top_k]

    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
        doc_topics=doc_topics,
        reranker=FakeRerankerPreferringOtherDocs(),
        rerank_candidates=6,
    )
    results = retriever.retrieve("what is the leave policy")
    assert all(chunk.source_file == "leave.pdf" for chunk in results)


def test_retrieve_logs_mode_and_mapped_document(caplog):
    import logging

    vector_store = FakeVectorStore(
        distances=[0.1],
        metadatas=[{"source_file": "leave.pdf", "page_number": 1, "chunk_index": 0}],
    )
    doc_topics = [DocTopic(topic="Leave Policy", link="https://example.com/leave", docs_filename="leave.pdf")]
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
        doc_topics=doc_topics,
    )
    with caplog.at_level(logging.INFO, logger="rag.retrieval.retriever"):
        retriever.retrieve("what is the leave policy")
    assert any(
        f"mode={MAPPED_DOCUMENT_MODE}" in record.message and "leave.pdf" in record.message
        for record in caplog.records
    )


def test_retrieve_logs_global_mode_when_no_topic_matches(caplog):
    import logging

    vector_store = FakeVectorStore(distances=[0.9])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
    )
    with caplog.at_level(logging.INFO, logger="rag.retrieval.retriever"):
        retriever.retrieve("totally unrelated question")
    assert any(f"mode={GLOBAL_MODE}" in record.message for record in caplog.records)


# --- MAPPED_DOCUMENT mode: PRIMARY + SECONDARY merge (retrieve() branching) -

# These test retrieve()'s merge logic in isolation from the real search
# machinery (dense/BM25/RRF/rerank, exercised elsewhere in this file)
# by stubbing out Retriever._search itself, per the four scenarios in
# the PRIMARY+SECONDARY design: primary always precedes secondary,
# secondary only fills what's left of top_k, and an empty primary
# falls back fully to GLOBAL_MODE rather than merging.


def _chunk(label: str) -> RetrievedChunk:
    return RetrievedChunk(text=label, source_file=label, page_number=1, relevance_score=1.0)


def _stub_search(primary_chunks: list, secondary_chunks: list):
    """Replaces Retriever._search so these tests exercise only
    retrieve()'s PRIMARY+SECONDARY merge/branching, independent of
    whatever MAPPED_DOCUMENT_PRIMARY_TOP_K production currently uses
    for the PRIMARY call's own `keep`: the PRIMARY call (metadata_filter
    pins a source_file) returns primary_chunks in full, as a given —
    exactly like the four PRIMARY/SECONDARY scenarios in the design
    spec, which describe primary/secondary as already-sized inputs to
    the merge, not something this stub should re-truncate. Any other
    call (SECONDARY, or the GLOBAL fallback when primary is empty) is
    truncated to `keep`, since respecting "fill only what's left of
    top_k" is exactly the SECONDARY behavior under test.
    """
    calls = []

    def fake_search(self, query, query_embedding, metadata_filter, candidate_count, keep=None, exclude_source_file=None):
        calls.append(
            {
                "metadata_filter": metadata_filter,
                "exclude_source_file": exclude_source_file,
                "keep": keep,
            }
        )
        if metadata_filter:
            return primary_chunks, primary_chunks
        keep = len(secondary_chunks) if keep is None else keep
        return secondary_chunks, secondary_chunks[:keep]

    return fake_search, calls


def _mapped_retriever(monkeypatch, primary_chunks, secondary_chunks, top_k):
    fake_search, calls = _stub_search(primary_chunks, secondary_chunks)
    monkeypatch.setattr(Retriever, "_search", fake_search)
    retriever = Retriever(
        vector_store=None,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=top_k,
        min_relevance_score=0.35,
        doc_topics=[DocTopic(topic="Doc A Policy", link="https://example.com/a", docs_filename="A")],
    )
    return retriever, calls


def test_merge_primary_then_secondary_fills_remaining_top_k(monkeypatch):
    primary = [_chunk("A1"), _chunk("A2")]
    secondary = [_chunk("B1"), _chunk("C1"), _chunk("B2")]
    retriever, _ = _mapped_retriever(monkeypatch, primary, secondary, top_k=4)

    results = retriever.retrieve("doc a policy")

    assert [chunk.source_file for chunk in results] == ["A1", "A2", "B1", "C1"]


def test_merge_primary_alone_fills_top_k_no_secondary_search(monkeypatch):
    primary = [_chunk("A1"), _chunk("A2"), _chunk("A3"), _chunk("A4")]
    secondary = [_chunk("B1"), _chunk("C1")]
    retriever, calls = _mapped_retriever(monkeypatch, primary, secondary, top_k=4)

    results = retriever.retrieve("doc a policy")

    assert [chunk.source_file for chunk in results] == ["A1", "A2", "A3", "A4"]
    # No leftover top_k slots -> the secondary search must never run.
    assert len(calls) == 1


def test_merge_single_primary_result_fills_rest_from_secondary(monkeypatch):
    primary = [_chunk("A1")]
    secondary = [_chunk("B1"), _chunk("C1"), _chunk("B2")]
    retriever, calls = _mapped_retriever(monkeypatch, primary, secondary, top_k=4)

    results = retriever.retrieve("doc a policy")

    assert [chunk.source_file for chunk in results] == ["A1", "B1", "C1", "B2"]
    assert calls[1]["exclude_source_file"] == "A"
    assert calls[1]["keep"] == 3


def test_merge_falls_back_to_global_mode_when_primary_is_empty(monkeypatch, caplog):
    import logging

    secondary = [_chunk("B1"), _chunk("C1")]
    retriever, calls = _mapped_retriever(monkeypatch, [], secondary, top_k=4)

    with caplog.at_level(logging.INFO, logger="rag.retrieval.retriever"):
        results = retriever.retrieve("doc a policy")

    # Falls back fully to GLOBAL_MODE (no primary+secondary merge): the
    # second _search call is a plain, unrestricted GLOBAL search, not
    # a SECONDARY one excluding the mapped document.
    assert [chunk.source_file for chunk in results] == ["B1", "C1"]
    assert len(calls) == 2
    assert calls[1]["exclude_source_file"] is None
    assert any(f"mode={GLOBAL_MODE}" in record.message for record in caplog.records)
