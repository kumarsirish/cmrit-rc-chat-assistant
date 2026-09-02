from pathlib import Path

from rag.document_loader import find_pdf_files, load_all_documents, load_pdf

POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies_docs"


def test_find_pdf_files_returns_only_pdfs():
    pdf_files = find_pdf_files(str(POLICIES_DIR))
    assert len(pdf_files) > 0
    for pdf_file in pdf_files:
        assert pdf_file.suffix == ".pdf"


def test_load_pdf_extracts_non_empty_pages():
    pdf_files = find_pdf_files(str(POLICIES_DIR))
    pages = load_pdf(pdf_files[0])
    assert len(pages) > 0
    for page in pages:
        assert page.text.strip() != ""
        assert page.source_file == pdf_files[0].name
        assert page.page_number >= 1


def test_load_all_documents_covers_every_pdf():
    pdf_files = find_pdf_files(str(POLICIES_DIR))
    pages = load_all_documents(str(POLICIES_DIR))
    source_files_seen = set(page.source_file for page in pages)
    expected_files = set(pdf_file.name for pdf_file in pdf_files)
    assert source_files_seen == expected_files
