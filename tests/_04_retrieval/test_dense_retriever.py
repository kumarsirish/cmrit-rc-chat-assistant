from rag._04_retrieval.dense_retriever import DenseRetriever


class FakeVectorStore:
    def __init__(self):
        self.last_call = None

    def similarity_search_with_relevance_scores(self, query_embedding, top_k, filter=None):
        self.last_call = {"query_embedding": query_embedding, "top_k": top_k, "filter": filter}
        return [("doc", 0.9)]


def test_search_passes_the_given_embedding_and_filter_through_to_the_vector_store():
    vector_store = FakeVectorStore()
    retriever = DenseRetriever(vector_store)

    results = retriever.search([1.0, 0.0], top_k=3, filter={"source_file": "a.pdf"})

    assert results == [("doc", 0.9)]
    assert vector_store.last_call == {
        "query_embedding": [1.0, 0.0],
        "top_k": 3,
        "filter": {"source_file": "a.pdf"},
    }
