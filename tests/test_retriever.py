from rag.retriever import Retriever


class FakeEmbeddingGenerator:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, distances, keyword_matches=None):
        self._distances = distances
        # keyword_matches: list of (text, source_file, page_number, embedding)
        self._keyword_matches = keyword_matches or []

    def query(self, query_embedding, top_k):
        documents = [f"chunk text {i}" for i in range(len(self._distances))]
        metadatas = [
            {"source_file": f"doc{i}.pdf", "page_number": i + 1}
            for i in range(len(self._distances))
        ]
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [self._distances],
        }

    def find_chunks_containing_any(self, keywords, limit):
        documents = [match[0] for match in self._keyword_matches]
        metadatas = [
            {"source_file": match[1], "page_number": match[2]}
            for match in self._keyword_matches
        ]
        embeddings = [match[3] for match in self._keyword_matches]
        return {"documents": documents, "metadatas": metadatas, "embeddings": embeddings}


def test_retrieve_filters_out_low_relevance_chunks():
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
    assert results[0].match_type == "semantic"


def test_retrieve_returns_empty_list_when_nothing_is_relevant_and_no_keyword_hits():
    vector_store = FakeVectorStore(distances=[0.8, 0.95])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
    )
    results = retriever.retrieve("unrelated question")
    assert results == []


def test_retrieve_falls_back_to_keyword_match_when_semantic_search_finds_nothing():
    # Semantic search finds nothing above threshold ...
    vector_store = FakeVectorStore(
        distances=[0.9],
        # ... but a literal keyword match exists, with an embedding
        # identical to the query embedding (cosine similarity 1.0).
        keyword_matches=[
            ("Leave Application through Darwinbox...", "leave.pdf", 2, [1.0, 0.0, 0.0]),
        ],
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=4,
        min_relevance_score=0.35,
    )
    results = retriever.retrieve("what is darwinbox")
    assert len(results) == 1
    assert results[0].source_file == "leave.pdf"
    assert results[0].match_type == "keyword"
    assert results[0].relevance_score == 1.0


def test_retrieve_keyword_fallback_returns_nothing_for_only_stopwords():
    vector_store = FakeVectorStore(distances=[0.9], keyword_matches=[])
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=FakeEmbeddingGenerator(),
        top_k=2,
        min_relevance_score=0.35,
    )
    results = retriever.retrieve("what is it")
    assert results == []
