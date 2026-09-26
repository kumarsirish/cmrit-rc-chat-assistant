from pathlib import Path

import pytest

from rag._01_ingestion.document_loader import (
    DocumentLoadError,
    find_pdf_files,
    load_all_documents,
    load_pdf,
)

POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "policies_docs"


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


def test_load_pdf_raises_for_a_file_that_is_not_a_real_pdf(tmp_path):
    bad_pdf = tmp_path / "garbage.pdf"
    bad_pdf.write_bytes(b"not a real pdf file at all")

    with pytest.raises(DocumentLoadError, match="Could not open/parse"):
        load_pdf(bad_pdf)


def test_load_all_documents_skips_an_unreadable_pdf_and_still_loads_the_rest(tmp_path):
    good_pdf_source = find_pdf_files(str(POLICIES_DIR))[0]
    (tmp_path / good_pdf_source.name).write_bytes(good_pdf_source.read_bytes())
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf file at all")

    pages = load_all_documents(str(tmp_path))

    source_files_seen = {page.source_file for page in pages}
    assert good_pdf_source.name in source_files_seen
    assert "corrupt.pdf" not in source_files_seen


def test_load_all_documents_raises_when_every_pdf_is_unreadable(tmp_path):
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf file at all")

    with pytest.raises(DocumentLoadError, match="No usable PDF pages"):
        load_all_documents(str(tmp_path))
