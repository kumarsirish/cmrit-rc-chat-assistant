"""Classic IR metrics for judging retrieval quality on its own, apart
from the LLM-judged generation metrics in test_judge.py.

Every function takes retrieved_sources — a ranked list of source_file
strings, most relevant first, exactly as Retriever.retrieve() returns
them via chunk.source_file — and relevant_sources, the ground-truth
set of source files that should have been retrieved for that question
(see the "relevant_sources" field in evaluation-question-answers.json).

Relevance here is binary and document-level (a source file either is
or isn't one of the right documents for the question), which matches
what a small policy-PDF test set can realistically be hand-labeled
with — no per-chunk relevance judgments are needed.
"""

import math


def recall_at_k(retrieved_sources: list, relevant_sources: set, k: int) -> float:
    """Fraction of relevant_sources that appear anywhere in the top k
    retrieved_sources. Returns 1.0 if relevant_sources is empty (an
    unanswerable question has nothing to find, so there's nothing to
    miss)."""
    if not relevant_sources:
        return 1.0
    top_k = set(retrieved_sources[:k])
    return len(top_k & relevant_sources) / len(relevant_sources)


def precision_at_k(retrieved_sources: list, relevant_sources: set, k: int) -> float:
    """Fraction of the top k retrieved_sources that are relevant.
    Returns 1.0 for an unanswerable question if nothing was retrieved
    (correctly finding nothing to retrieve), 0.0 if something was
    retrieved anyway (a false positive)."""
    top_k = retrieved_sources[:k]
    if not top_k:
        return 1.0 if not relevant_sources else 0.0
    hits = sum(1 for source in top_k if source in relevant_sources)
    return hits / len(top_k)


def reciprocal_rank(retrieved_sources: list, relevant_sources: set) -> float:
    """1 / (rank of the first relevant source), or 0.0 if none of
    retrieved_sources is relevant. Returns 1.0 for an unanswerable
    question if nothing was retrieved."""
    if not relevant_sources:
        return 1.0 if not retrieved_sources else 0.0
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant_sources:
            return 1.0 / rank
    return 0.0


def _unique_in_rank_order(sources: list) -> list:
    """De-duplicates sources to one entry per distinct document, kept
    at its first (best) rank. retrieved_sources is a per-chunk list —
    the same source_file legitimately appears more than once when
    several of the top-k chunks come from the same document (e.g. a
    mapped-document query, or one document dominating a global
    search). Relevance here is document-level, so a document must
    contribute at most one ideal-DCG slot and at most one actual-DCG
    term, or DCG can outscore IDCG for the trivial reason that one
    relevant document happened to fill multiple ranks."""
    seen = set()
    unique = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


def ndcg_at_k(retrieved_sources: list, relevant_sources: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, with binary
    (0/1) relevance per source file. Rewards relevant sources ranked
    higher more than the same sources ranked lower, unlike recall/
    precision which only look at *whether* they're in the top k."""
    if not relevant_sources:
        return 1.0 if not retrieved_sources[:k] else 0.0

    top_k = _unique_in_rank_order(retrieved_sources)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, source in enumerate(top_k, start=1) if source in relevant_sources
    )
    ideal_hits = min(len(relevant_sources), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
