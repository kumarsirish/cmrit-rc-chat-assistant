"""Generic dense + BM25 hybrid retrieval, fused by Reciprocal Rank
Fusion (RRF).

This module knows nothing about topics, mapped documents, or the
retrieval policy (MAPPED_DOCUMENT/PRIMARY/SECONDARY/GLOBAL) — see
retrieval_service.py for that. It just takes a dense retriever, a
BM25 index, and a query, and returns one fused, ranked candidate list.

RRF combines by rank rather than raw score, since cosine similarity
and BM25 scores are on different scales and can't be compared or
added directly.
"""

import operator
from dataclasses import dataclass

# Constant from the original RRF paper. Damps how much a #1-vs-#2 rank
# difference matters, so it doesn't need tuning per corpus size.
RRF_K = 60


@dataclass
class FusedCandidate:
    """One chunk after fusion, with both raw scores and ranks kept
    around for logging/debugging, plus the RRF score used for
    ordering."""

    text: str
    metadata: dict
    dense_score: float = None
    dense_rank: int = None
    lexical_score: float = None
    lexical_rank: int = None
    fused_score: float = 0.0


def _document_text(document) -> str:
    # Dense results are langchain Documents (.page_content); BM25
    # results are BM25Document (.text).
    return document.page_content if hasattr(document, "page_content") else document.text


def _chunk_key(metadata: dict) -> tuple:
    return (
        metadata.get("source_file"),
        metadata.get("page_number"),
        metadata.get("chunk_index", 0),
    )


def _add_results(candidates: dict, results: list, score_field: str, rank_field: str, weight: float) -> None:
    for rank, (document, score) in enumerate(results):
        key = _chunk_key(document.metadata)
        candidate = candidates.get(key)
        if candidate is None:
            candidate = FusedCandidate(text=_document_text(document), metadata=document.metadata)
            candidates[key] = candidate
        setattr(candidate, score_field, score)
        setattr(candidate, rank_field, rank)
        candidate.fused_score += weight / (RRF_K + rank + 1)


def fuse(dense_results: list, lexical_results: list, dense_weight: float = 0.5, lexical_weight: float = 0.5) -> list:
    """Combines dense_results and lexical_results (each a list of
    (document, score), most relevant first) into one list of
    FusedCandidate, sorted by fused_score descending.

    dense_weight/lexical_weight scale each list's contribution to the
    fused score. Equal weights (the default) means neither search
    method is trusted more than the other by default.
    """
    candidates = {}
    _add_results(candidates, dense_results, "dense_score", "dense_rank", dense_weight)
    _add_results(candidates, lexical_results, "lexical_score", "lexical_rank", lexical_weight)

    fused = list(candidates.values())
    fused.sort(key=operator.attrgetter("fused_score"), reverse=True)
    return fused
