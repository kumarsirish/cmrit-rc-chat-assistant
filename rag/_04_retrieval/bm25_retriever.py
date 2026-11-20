"""BM25 lexical retrieval.

Dense embedding search can miss exact keyword matches (acronyms,
policy codes, proper nouns). BM25 scores chunks by term frequency
against the query's literal tokens, which covers that gap.

This stays custom instead of using langchain_community's BM25Retriever
for two reasons: BM25Retriever has no way to filter by metadata after
scoring (needed for PRIMARY/SECONDARY/GLOBAL scoping), and it has no
token-overlap gate, so it can return a "match" with a near-zero score
and no real lexical overlap. Building one BM25Retriever per source_file
instead would also change the IDF/avgdl statistics for every term,
which is a different retrieval model, not just a filtering change.

The whole corpus is small (a few hundred/thousand chunks), so the
index is just an in-memory rank_bm25.BM25Okapi built once from every
chunk in the Chroma collection, rather than persisted separately.
"""

import operator
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "whom", "which", "how", "why", "when", "where",
    "do", "does", "did", "doing", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "for", "of", "to", "in", "on", "at", "by", "with", "about", "as",
    "and", "or", "but", "if", "so", "this", "that", "these", "those",
    "tell", "explain", "please",
}


def _tokenize(text: str) -> list:
    # Stopwords removed so a query like "what is the capital of
    # France?" doesn't share "is"/"the"/"of" with nearly every chunk.
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOPWORDS]


def _search_text(document: "BM25Document") -> str:
    keywords = (document.metadata or {}).get("keywords", "")
    return f"{document.text} {keywords}" if keywords else document.text


def _matches_filter(metadata: dict, metadata_filter: dict) -> bool:
    """Mirrors the subset of Chroma's `where` operators retrieval_service.py
    actually builds: a plain value means exact match, {"$in": [...]}
    means membership (used for PRIMARY search's possibly-multi-file
    scoped_filter)."""
    if not metadata_filter:
        return True
    for key, value in metadata_filter.items():
        actual = metadata.get(key)
        if isinstance(value, dict) and "$in" in value:
            if actual not in value["$in"]:
                return False
        elif actual != value:
            return False
    return True


@dataclass
class BM25Document:
    """One chunk as seen by the BM25 index."""

    text: str
    metadata: dict


class BM25Index:
    """In-memory BM25 index over every chunk in the vector store."""

    def __init__(self, documents: list):
        self._documents = documents
        # Tokenize text + any curated keywords for the chunk's topic
        # (see helper_scripts/ingest.py's search_text()), not text alone, so
        # BM25 can match a query term (e.g. "Director") a chunk's own
        # prose never uses. Only the tokens fed to BM25 are affected —
        # document.text itself (what gets shown to the LLM/cited)
        # never changes.
        corpus = [_tokenize(_search_text(document)) for document in documents]
        self._token_sets = [set(tokens) for tokens in corpus]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_vector_store(cls, vector_store) -> "BM25Index":
        """Builds the index from every chunk currently stored in
        vector_store. Called once at startup."""
        documents = [
            BM25Document(text=text, metadata=metadata)
            for text, metadata in vector_store.get_all_documents()
        ]
        return cls(documents)

    def search(self, query: str, top_k: int, metadata_filter: dict = None) -> list:
        """Returns up to top_k (BM25Document, score) pairs whose text
        shares at least one token with query, most relevant first,
        optionally restricted by metadata_filter.
        """
        if not self._bm25 or not self._documents:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        # A token that appears in every document of a small corpus can
        # give BM25 a score near/below zero even for a real match, so
        # token overlap (not the raw score) decides candidacy; the
        # score is only used to rank the candidates that qualify.
        scores = self._bm25.get_scores(list(query_tokens))
        candidates = [
            (document, score)
            for document, score, token_set in zip(self._documents, scores, self._token_sets)
            if token_set & query_tokens and _matches_filter(document.metadata, metadata_filter)
        ]
        candidates.sort(key=operator.itemgetter(1), reverse=True)
        return candidates[:top_k]
