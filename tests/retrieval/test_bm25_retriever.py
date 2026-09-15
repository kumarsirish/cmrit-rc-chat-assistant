from rag.retrieval.bm25_retriever import BM25Document, BM25Index


def _index(entries):
    """entries: list of (text, metadata)."""
    return BM25Index([BM25Document(text=text, metadata=metadata) for text, metadata in entries])


def test_search_finds_literal_token_match():
    index = _index(
        [
            ("Leave application through Darwinbox portal.", {"source_file": "leave.pdf"}),
            ("Probation completion process for new staff.", {"source_file": "probation.pdf"}),
        ]
    )
    results = index.search("what is darwinbox", top_k=5)
    assert len(results) == 1
    assert results[0][0].metadata["source_file"] == "leave.pdf"


def test_search_returns_empty_for_no_token_overlap():
    index = _index([("Leave application through Darwinbox portal.", {"source_file": "leave.pdf"})])
    results = index.search("completely unrelated query", top_k=5)
    assert results == []


def test_search_returns_empty_for_query_with_no_tokens():
    index = _index([("Some policy text.", {"source_file": "doc.pdf"})])
    assert index.search("???", top_k=5) == []


def test_search_applies_metadata_filter():
    index = _index(
        [
            ("Casual leave policy details.", {"source_file": "a.pdf", "topic": "leave"}),
            ("Casual leave policy details.", {"source_file": "b.pdf", "topic": "other"}),
        ]
    )
    results = index.search("casual leave", top_k=5, metadata_filter={"topic": "leave"})
    assert len(results) == 1
    assert results[0][0].metadata["source_file"] == "a.pdf"


def test_search_ranks_more_frequent_term_matches_higher():
    # A third, unrelated document keeps "mentoring" from appearing in
    # every document in the corpus — if it did, BM25's IDF term would
    # go negative and (correctly, if unintuitively) rank *fewer*
    # mentions higher, which isn't the behavior under test here.
    index = _index(
        [
            ("mentoring mentoring mentoring process overview.", {"source_file": "high.pdf"}),
            ("a brief mention of mentoring once.", {"source_file": "low.pdf"}),
            ("probation completion process for new staff.", {"source_file": "unrelated.pdf"}),
        ]
    )
    results = index.search("mentoring", top_k=5)
    assert [document.metadata["source_file"] for document, _ in results] == ["high.pdf", "low.pdf"]


def test_empty_index_returns_no_results():
    index = _index([])
    assert index.search("anything", top_k=5) == []
