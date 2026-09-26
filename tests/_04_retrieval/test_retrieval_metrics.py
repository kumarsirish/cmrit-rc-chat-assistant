from rag._07_evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)


def test_recall_at_k_counts_all_relevant_sources_found_in_top_k():
    retrieved = ["a.pdf", "b.pdf", "c.pdf"]
    relevant = {"a.pdf", "c.pdf", "d.pdf"}
    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3


def test_recall_at_k_ignores_hits_outside_k():
    retrieved = ["b.pdf", "a.pdf"]
    relevant = {"a.pdf"}
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_at_k_is_perfect_for_unanswerable_question():
    assert recall_at_k(["a.pdf"], set(), k=3) == 1.0


def test_precision_at_k_counts_relevant_fraction_of_top_k():
    retrieved = ["a.pdf", "x.pdf", "c.pdf"]
    relevant = {"a.pdf", "c.pdf"}
    assert precision_at_k(retrieved, relevant, k=3) == 2 / 3


def test_precision_at_k_penalizes_retrieving_anything_for_unanswerable_question():
    assert precision_at_k(["x.pdf"], set(), k=3) == 0.0
    assert precision_at_k([], set(), k=3) == 1.0


def test_reciprocal_rank_uses_first_relevant_hit():
    retrieved = ["x.pdf", "a.pdf", "b.pdf"]
    relevant = {"a.pdf", "b.pdf"}
    assert reciprocal_rank(retrieved, relevant) == 1 / 2


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_found():
    assert reciprocal_rank(["x.pdf"], {"a.pdf"}) == 0.0


def test_reciprocal_rank_for_unanswerable_question():
    assert reciprocal_rank([], set()) == 1.0
    assert reciprocal_rank(["x.pdf"], set()) == 0.0


def test_ndcg_rewards_relevant_source_ranked_first_over_second():
    relevant = {"a.pdf"}
    assert ndcg_at_k(["a.pdf", "x.pdf"], relevant, k=2) == 1.0
    score_when_second = ndcg_at_k(["x.pdf", "a.pdf"], relevant, k=2)
    assert 0 < score_when_second < 1.0


def test_ndcg_is_zero_when_nothing_relevant_is_found():
    assert ndcg_at_k(["x.pdf", "y.pdf"], {"a.pdf"}, k=2) == 0.0


def test_ndcg_for_unanswerable_question():
    assert ndcg_at_k([], set(), k=2) == 1.0
    assert ndcg_at_k(["x.pdf"], set(), k=2) == 0.0


def test_ndcg_never_exceeds_one_when_one_relevant_document_fills_every_rank():
    # Regression test: retrieved_sources is a per-chunk list, so the
    # single relevant document filling all k top-k chunk slots (e.g.
    # three chunks from the one relevant PDF) must not let DCG sum a
    # separate gain term per chunk while IDCG only awards one ideal
    # slot for that one relevant document — that mismatch is what
    # previously pushed NDCG above 1.0 (observed: 2.1309297535714578
    # for retrieved=["a.pdf", "a.pdf", "a.pdf"], relevant={"a.pdf"}).
    retrieved = ["a.pdf", "a.pdf", "a.pdf"]
    relevant = {"a.pdf"}
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0


def test_ndcg_deduplicates_repeated_chunks_from_the_same_document():
    # Two distinct relevant documents, but the first one occupies two
    # of the three chunk slots. Only one ideal slot should reward it
    # (rank-1 gain), matching IDCG's two ideal slots for the two
    # distinct relevant documents overall.
    retrieved = ["a.pdf", "a.pdf", "b.pdf"]
    relevant = {"a.pdf", "b.pdf"}
    import math

    expected_dcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(retrieved, relevant, k=3) == expected_dcg / expected_idcg
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0
