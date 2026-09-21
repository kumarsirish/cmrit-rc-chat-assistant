import pytest

from rag.indexing.vector_store import VectorStore, VectorStoreError


def _make_store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path), collection_name="test_collection")


def test_similarity_search_wraps_a_chroma_failure(tmp_path, monkeypatch):
    store = _make_store(tmp_path)

    def _raise(*args, **kwargs):
        raise RuntimeError("collection is corrupt")

    monkeypatch.setattr(store._store, "similarity_search_by_vector_with_relevance_scores", _raise)

    with pytest.raises(VectorStoreError, match="Chroma similarity search failed"):
        store.similarity_search_with_relevance_scores([1.0, 0.0, 0.0], top_k=3)
