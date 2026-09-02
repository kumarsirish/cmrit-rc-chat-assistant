from rag.vector_store import VectorStore


def _make_store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path), collection_name="test_collection")


def test_find_chunks_containing_any_is_case_insensitive(tmp_path):
    store = _make_store(tmp_path)
    store.add_chunks(
        ids=["chunk-1"],
        texts=["Dr. Sabitha Ramamurthy is the President, CMRJT."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"source_file": "handbook.pdf", "page_number": 6}],
    )

    # Query keyword is lowercase; the source text has "President" capitalized.
    results = store.find_chunks_containing_any(["president"], limit=10)

    assert len(results["documents"]) == 1
    assert "President" in results["documents"][0]
    assert results["metadatas"][0]["page_number"] == 6


def test_find_chunks_containing_any_returns_empty_for_no_keywords(tmp_path):
    store = _make_store(tmp_path)
    results = store.find_chunks_containing_any([], limit=10)
    assert results == {"documents": [], "metadatas": [], "embeddings": []}


def test_find_chunks_containing_any_matches_across_multiple_keywords(tmp_path):
    store = _make_store(tmp_path)
    store.add_chunks(
        ids=["chunk-1", "chunk-2", "chunk-3"],
        texts=[
            "Dr. Sabitha Ramamurthy is the President, CMRJT.",
            "Sri. K C Ramamurthy is the Chairman, CMRGI.",
            "This chunk mentions neither term.",
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        metadatas=[
            {"source_file": "handbook.pdf", "page_number": 6},
            {"source_file": "handbook.pdf", "page_number": 6},
            {"source_file": "handbook.pdf", "page_number": 7},
        ],
    )

    results = store.find_chunks_containing_any(["president", "chairman"], limit=10)

    assert len(results["documents"]) == 2
