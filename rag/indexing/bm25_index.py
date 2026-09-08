"""Step 4a: BM25 lexical retrieval.

Dense embedding search (see vector_store.py) captures semantic
similarity but can miss exact keyword matches — acronyms, policy
codes, proper nouns — that the embedding model has too little signal
to place correctly. BM25 scores chunks by term frequency against the
query's literal tokens, which is exactly the signal dense search is
weak on.

The whole corpus is small (a few hundred/thousand chunks), so the
index is just an in-memory rank_bm25.BM25Okapi built once from every
chunk in the Chroma collection (see BM25Index.from_vector_store),
rather than anything persisted separately from the vector store.
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from rag.retrieval.keyword_extraction import _STOPWORDS

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    # Stopwords excluded (same list keyword_extraction.py uses): left
    # in, a query like "What is the capital of France?" shares "is"/
    # "the"/"of" with nearly every chunk in the corpus, which BM25
    # would then score as a genuine lexical match — exactly the false
    # positive a completely unrelated question must NOT produce, since
    # a lexical hit bypasses the dense relevance threshold downstream
    # (see retriever.py).
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOPWORDS]


def _matches_filter(metadata: dict, metadata_filter: dict) -> bool:
    if not metadata_filter:
        return True
    return all(metadata.get(key) == value for key, value in metadata_filter.items())


@dataclass
class BM25Document:
    """One chunk as seen by the BM25 index: just enough to score and
    identify it, mirroring the shape of a langchain Document (.metadata)
    without depending on langchain."""

    text: str
    metadata: dict


class BM25Index:
    """In-memory BM25 index over every chunk in the vector store."""

    def __init__(self, documents: list):
        self._documents = documents
        corpus = [_tokenize(document.text) for document in documents]
        self._token_sets = [set(tokens) for tokens in corpus]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_vector_store(cls, vector_store) -> "BM25Index":
        """Builds the index from every chunk currently stored in
        vector_store. Called once at startup; the app is read-only
        against the index between ingestion runs."""
        documents = [
            BM25Document(text=text, metadata=metadata)
            for text, metadata in vector_store.get_all_documents()
        ]
        return cls(documents)

    def search(self, query: str, top_k: int, metadata_filter: dict = None) -> list:
        """Returns up to top_k (BM25Document, score) pairs whose text
        shares at least one token with query, most relevant first,
        optionally restricted to documents whose metadata matches every
        key/value in metadata_filter.
        """
        if not self._bm25 or not self._documents:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        # BM25's IDF term goes to 0/negative for a token that appears
        # in every document of a small corpus (trivially true for a
        # 1-2 document test fixture, and possible in production for a
        # near-universal word) — a plain score > 0 filter would wrongly
        # drop a real, literal match in that case. Token overlap
        # decides *candidacy*; the (possibly negative) BM25 score is
        # only used to *rank* the candidates that qualify.
        scores = self._bm25.get_scores(list(query_tokens))
        candidates = [
            (document, score)
            for document, score, token_set in zip(self._documents, scores, self._token_sets)
            if token_set & query_tokens and _matches_filter(document.metadata, metadata_filter)
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:top_k]
