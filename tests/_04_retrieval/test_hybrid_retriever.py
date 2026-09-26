from dataclasses import dataclass, field

from rag._04_retrieval.hybrid_retriever import fuse


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


def _doc(source_file, page_number=1):
    return FakeDocument(page_content=f"text {source_file}", metadata={"source_file": source_file, "page_number": page_number})


def test_fuse_combines_dense_and_lexical_hits_for_the_same_chunk():
    dense_results = [(_doc("a.pdf"), 0.9)]
    lexical_results = [(_doc("a.pdf"), 5.0)]
    fused = fuse(dense_results, lexical_results)
    assert len(fused) == 1
    assert fused[0].dense_score == 0.9
    assert fused[0].lexical_score == 5.0
    assert fused[0].dense_rank == 0
    assert fused[0].lexical_rank == 0


def test_fuse_ranks_a_chunk_found_by_both_lists_above_one_found_by_a_single_list():
    dense_results = [(_doc("a.pdf"), 0.9), (_doc("b.pdf"), 0.8)]
    lexical_results = [(_doc("a.pdf"), 5.0)]
    fused = fuse(dense_results, lexical_results)
    assert fused[0].metadata["source_file"] == "a.pdf"


def test_fuse_keeps_a_dense_only_and_a_lexical_only_chunk_both():
    dense_results = [(_doc("a.pdf"), 0.9)]
    lexical_results = [(_doc("b.pdf"), 5.0)]
    fused = fuse(dense_results, lexical_results)
    sources = {candidate.metadata["source_file"] for candidate in fused}
    assert sources == {"a.pdf", "b.pdf"}


def test_fuse_weights_change_which_list_dominates_the_fused_order():
    # Same rank (0) in each list, so equal weights tie; weighting
    # lexical higher must move it ahead.
    dense_results = [(_doc("dense.pdf"), 0.9)]
    lexical_results = [(_doc("lexical.pdf"), 5.0)]
    fused = fuse(dense_results, lexical_results, dense_weight=0.1, lexical_weight=0.9)
    assert fused[0].metadata["source_file"] == "lexical.pdf"
