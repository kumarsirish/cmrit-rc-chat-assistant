from rag.chunker import chunk_documents, chunk_text
from rag.document_loader import PageDocument

SEPARATORS = ["\n\n", "\n", ". ", " "]


def test_short_text_is_returned_as_a_single_chunk():
    text = "This is a short policy sentence."
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20, separators=SEPARATORS)
    assert chunks == [text]


def test_long_text_is_split_into_multiple_chunks():
    text = (
        "First sentence about leave policy. "
        "Second sentence about approval process. "
        "Third sentence about documentation requirements. "
        "Fourth sentence about notice periods and manager sign-off. "
    ) * 5
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=30, separators=SEPARATORS)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200 + 30  # allow the small overlap carried forward


def test_urls_are_not_split_or_mangled():
    text = (
        "For more details visit https://cmrit.ac.in/policies/leave.pdf and "
        "read the full document. Contact hr@cmrit.ac.in for questions. "
        "See also www.cmrit.ac.in/staff for the staff portal."
    )
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10, separators=SEPARATORS)
    combined = " ".join(chunks)
    assert "https://cmrit.ac.in/policies/leave.pdf" in combined
    assert "www.cmrit.ac.in/staff" in combined


def test_consecutive_chunks_overlap():
    text = "Sentence one is here. " * 30
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20, separators=SEPARATORS)
    assert len(chunks) > 1
    first_tail = chunks[0][-15:]
    assert first_tail in chunks[1]


def test_blank_line_with_stray_whitespace_still_breaks_a_paragraph():
    # PDF extraction sometimes leaves a stray space on an otherwise
    # blank line ("\n \n" instead of "\n\n"), which must still count
    # as a paragraph break so unrelated sections don't get merged into
    # the same chunk. Padding forces a split to actually be needed.
    section_a = "Section A is about program TYT, also called Touch Your TOES. " * 3
    section_b = "Section B is about program TYL, also called Tie Your LACES. " * 3
    text = section_a + "\n \n" + section_b  # stray-space blank line

    chunks = chunk_text(text, chunk_size=len(section_a) + 10, chunk_overlap=0, separators=SEPARATORS)

    assert len(chunks) == 2
    assert "TYT" in chunks[0] and "TYL" not in chunks[0]
    assert "TYL" in chunks[1] and "TYT" not in chunks[1]


def test_chunk_documents_preserves_source_metadata():
    pages = [
        PageDocument(text="Policy about leave applications.", source_file="leave.pdf", page_number=1),
        PageDocument(text="Policy about probation completion.", source_file="probation.pdf", page_number=3),
    ]
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=20, separators=SEPARATORS)
    assert len(chunks) == 2
    assert chunks[0].source_file == "leave.pdf"
    assert chunks[0].page_number == 1
    assert chunks[1].source_file == "probation.pdf"
    assert chunks[1].page_number == 3
