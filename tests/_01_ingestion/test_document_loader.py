from pathlib import Path

import pytest

from rag._01_ingestion.document_loader import (
    DocumentLoadError,
    find_document_files,
    load_all_documents,
    load_docx,
    load_pdf,
    load_txt,
    load_xlsx,
    save_uploaded_document,
)

POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_documents"


def test_find_document_files_returns_only_supported_extensions(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.docx").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    (tmp_path / "d.xlsx").write_bytes(b"x")
    (tmp_path / "e.unsupported").write_bytes(b"x")

    found = find_document_files(str(tmp_path))
    assert {path.name for path in found} == {"a.pdf", "b.docx", "c.txt", "d.xlsx"}


def test_find_document_files_excludes_archived_subfolder(tmp_path):
    (tmp_path / "active.txt").write_text("kept")
    archived_dir = tmp_path / "_archived"
    archived_dir.mkdir()
    (archived_dir / "old.txt").write_text("superseded")

    found = find_document_files(str(tmp_path), archived_subfolder_name="_archived")
    assert {path.name for path in found} == {"active.txt"}


def test_load_pdf_extracts_non_empty_pages():
    pdf_files = find_document_files(str(POLICIES_DIR))
    pdf_files = [path for path in pdf_files if path.suffix == ".pdf"]
    pages = load_pdf(pdf_files[0])
    assert len(pages) > 0
    for page in pages:
        assert page.text.strip() != ""
        assert page.source_file == pdf_files[0].name
        assert page.page_number >= 1


def test_load_docx_returns_one_page_with_all_paragraph_text(tmp_path):
    import docx

    docx_path = tmp_path / "notice.docx"
    document = docx.Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    document.save(str(docx_path))

    pages = load_docx(docx_path)
    assert len(pages) == 1
    assert pages[0].source_file == "notice.docx"
    assert pages[0].page_number == 1
    assert "First paragraph." in pages[0].text
    assert "Second paragraph." in pages[0].text


def test_load_docx_raises_for_empty_document(tmp_path):
    import docx

    docx_path = tmp_path / "empty.docx"
    docx.Document().save(str(docx_path))

    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_docx(docx_path)


def test_load_txt_returns_one_page(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Plain text content.")

    pages = load_txt(txt_path)
    assert len(pages) == 1
    assert pages[0].text == "Plain text content."
    assert pages[0].source_file == "notes.txt"
    assert pages[0].page_number == 1


def test_load_txt_raises_for_empty_file(tmp_path):
    txt_path = tmp_path / "empty.txt"
    txt_path.write_text("   ")

    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_txt(txt_path)


def test_load_xlsx_returns_one_page_per_sheet(tmp_path):
    import openpyxl

    xlsx_path = tmp_path / "data.xlsx"
    workbook = openpyxl.Workbook()
    first_sheet = workbook.active
    first_sheet.title = "Sheet1"
    first_sheet.append(["Policy", "Applies to everyone."])
    second_sheet = workbook.create_sheet("Sheet2")
    second_sheet.append(["Another note."])
    workbook.save(str(xlsx_path))

    pages = load_xlsx(xlsx_path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Applies to everyone." in pages[0].text
    assert pages[1].page_number == 2
    assert "Another note." in pages[1].text


def test_load_xlsx_raises_when_every_sheet_is_empty(tmp_path):
    import openpyxl

    xlsx_path = tmp_path / "blank.xlsx"
    openpyxl.Workbook().save(str(xlsx_path))

    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_xlsx(xlsx_path)


def test_load_all_documents_covers_every_document(tmp_path):
    import docx

    (tmp_path / "notes.txt").write_text("Some notes.")
    document = docx.Document()
    document.add_paragraph("Some docx content.")
    document.save(str(tmp_path / "notice.docx"))

    pages = load_all_documents(str(tmp_path))
    source_files_seen = {page.source_file for page in pages}
    assert source_files_seen == {"notes.txt", "notice.docx"}


def test_load_all_documents_skips_archived_files(tmp_path):
    (tmp_path / "active.txt").write_text("kept")
    archived_dir = tmp_path / "_archived"
    archived_dir.mkdir()
    (archived_dir / "old.txt").write_text("superseded")

    pages = load_all_documents(str(tmp_path), archived_subfolder_name="_archived")
    source_files_seen = {page.source_file for page in pages}
    assert source_files_seen == {"active.txt"}


def test_load_pdf_raises_for_a_file_that_is_not_a_real_pdf(tmp_path):
    bad_pdf = tmp_path / "garbage.pdf"
    bad_pdf.write_bytes(b"not a real pdf file at all")

    with pytest.raises(DocumentLoadError, match="Could not open/parse"):
        load_pdf(bad_pdf)


def test_load_all_documents_skips_an_unreadable_pdf_and_still_loads_the_rest(tmp_path):
    pdf_files = find_document_files(str(POLICIES_DIR))
    good_pdf_source = [path for path in pdf_files if path.suffix == ".pdf"][0]
    (tmp_path / good_pdf_source.name).write_bytes(good_pdf_source.read_bytes())
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf file at all")

    pages = load_all_documents(str(tmp_path))

    source_files_seen = {page.source_file for page in pages}
    assert good_pdf_source.name in source_files_seen
    assert "corrupt.pdf" not in source_files_seen


def test_load_all_documents_raises_when_every_document_is_unreadable(tmp_path):
    (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf file at all")

    with pytest.raises(DocumentLoadError, match="No usable document pages"):
        load_all_documents(str(tmp_path))


def test_save_uploaded_document_writes_under_sanitized_topic_folder(tmp_path):
    saved_path = save_uploaded_document(str(tmp_path), "Leave Policy", "leave.docx", b"content")
    assert saved_path == tmp_path / "Leave Policy" / "leave.docx"
    assert saved_path.read_bytes() == b"content"


def test_save_uploaded_document_strips_path_traversal_from_filename(tmp_path):
    saved_path = save_uploaded_document(str(tmp_path), "Leave Policy", "../../evil.txt", b"x")
    assert saved_path == tmp_path / "Leave Policy" / "evil.txt"
