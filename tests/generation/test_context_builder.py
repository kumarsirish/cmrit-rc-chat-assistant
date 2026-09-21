from rag.generation.context_builder import build
from rag.retrieval.retrieval_service import RetrievedChunk


def _chunk(**overrides):
    defaults = dict(
        text="Staff must apply for leave at least 3 days in advance.",
        source_file="leave.pdf",
        page_number=2,
        relevance_score=0.8,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_chunks_are_numbered_in_the_given_order():
    chunks = [
        _chunk(text="First chunk.", source_file="a.pdf", page_number=1),
        _chunk(text="Second chunk.", source_file="b.pdf", page_number=1),
    ]
    context = build(chunks)
    assert context.index("[Source 1: a.pdf") < context.index("[Source 2: b.pdf")
    assert "First chunk." in context
    assert "Second chunk." in context


def test_single_page_chunk_uses_singular_page_label():
    context = build([_chunk(page_number=4, page_end=4)])
    assert "page 4]" in context
    assert "pages" not in context


def test_multi_page_chunk_uses_a_page_range_label():
    context = build([_chunk(page_number=4, page_end=5)])
    assert "pages 4-5]" in context


def test_section_title_is_included_when_present():
    context = build([_chunk(section_title="Leave Eligibility")])
    assert 'section: "Leave Eligibility"' in context


def test_no_section_note_when_section_title_is_empty():
    context = build([_chunk(section_title="")])
    assert "section:" not in context


def test_chunk_text_is_preserved_verbatim():
    context = build([_chunk(text="Exact wording must survive formatting.")])
    assert "Exact wording must survive formatting." in context
