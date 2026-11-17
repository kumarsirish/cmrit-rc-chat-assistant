from pathlib import Path

import pytest

from rag._01_ingestion.document_loader import (
    DocumentLoadError,
    extract_zip_archive,
    find_document_files,
    list_categories,
    load_all_documents,
    load_docx,
    load_pdf,
    load_pptx,
    load_txt,
    load_xlsx,
    save_uploaded_document,
)

POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_documents"


def test_list_categories_lists_subfolders_sorted(tmp_path):
    (tmp_path / "Quality").mkdir()
    (tmp_path / "Leave Policy").mkdir()
    (tmp_path / "not_a_dir.pdf").write_bytes(b"x")

    assert list_categories(str(tmp_path)) == ["Leave Policy", "Quality"]


def test_list_categories_returns_empty_list_when_directory_missing(tmp_path):
    assert list_categories(str(tmp_path / "missing")) == []


def test_find_document_files_returns_only_supported_extensions(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.docx").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    (tmp_path / "d.xlsx").write_bytes(b"x")
    (tmp_path / "f.pptx").write_bytes(b"x")
    (tmp_path / "e.unsupported").write_bytes(b"x")

    found = find_document_files(str(tmp_path))
    assert {path.name for path in found} == {"a.pdf", "b.docx", "c.txt", "d.xlsx", "f.pptx"}


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


def test_load_docx_extracts_table_only_content(tmp_path):
    """A docx with no body paragraphs at all, only a table (e.g. a
    checklist laid out as a table) must still extract its content —
    see the "Checklist for RC Submission.docx" investigation, where
    document.paragraphs alone was empty."""
    import docx

    docx_path = tmp_path / "checklist.docx"
    document = docx.Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "S.No."
    table.cell(0, 1).text = "Checklist item"
    table.cell(1, 0).text = "1."
    table.cell(1, 1).text = "Follow the submission format."
    document.save(str(docx_path))

    pages = load_docx(docx_path)
    assert len(pages) == 1
    assert "Checklist item" in pages[0].text
    assert "Follow the submission format." in pages[0].text


def test_load_docx_extracts_paragraphs_and_tables_together(tmp_path):
    import docx

    docx_path = tmp_path / "mixed.docx"
    document = docx.Document()
    document.add_paragraph("Intro paragraph.")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Table content."
    document.save(str(docx_path))

    pages = load_docx(docx_path)
    assert "Intro paragraph." in pages[0].text
    assert "Table content." in pages[0].text


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


def test_load_pptx_returns_one_page_per_slide_with_text(tmp_path):
    from pptx import Presentation

    pptx_path = tmp_path / "deck.pptx"
    presentation = Presentation()
    blank_layout = presentation.slide_layouts[6]

    first_slide = presentation.slides.add_slide(blank_layout)
    first_box = first_slide.shapes.add_textbox(0, 0, 100, 100)
    first_box.text_frame.text = "First slide content."

    second_slide = presentation.slides.add_slide(blank_layout)
    second_box = second_slide.shapes.add_textbox(0, 0, 100, 100)
    second_box.text_frame.text = "Second slide content."

    presentation.save(str(pptx_path))

    pages = load_pptx(pptx_path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].text == "First slide content."
    assert pages[1].page_number == 2
    assert pages[1].text == "Second slide content."
    assert pages[0].source_file == "deck.pptx"


def test_load_pptx_skips_slides_with_no_text(tmp_path):
    from pptx import Presentation

    pptx_path = tmp_path / "deck.pptx"
    presentation = Presentation()
    blank_layout = presentation.slide_layouts[6]

    presentation.slides.add_slide(blank_layout)  # no text at all

    text_slide = presentation.slides.add_slide(blank_layout)
    text_box = text_slide.shapes.add_textbox(0, 0, 100, 100)
    text_box.text_frame.text = "Has content."

    presentation.save(str(pptx_path))

    pages = load_pptx(pptx_path)
    assert len(pages) == 1
    assert pages[0].page_number == 2
    assert pages[0].text == "Has content."


def test_load_pptx_raises_when_every_slide_is_empty(tmp_path):
    from pptx import Presentation

    pptx_path = tmp_path / "blank.pptx"
    presentation = Presentation()
    presentation.slides.add_slide(presentation.slide_layouts[6])
    presentation.save(str(pptx_path))

    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_pptx(pptx_path)


def test_load_pptx_raises_for_a_file_that_is_not_a_real_pptx(tmp_path):
    bad_pptx = tmp_path / "garbage.pptx"
    bad_pptx.write_bytes(b"not a real pptx file at all")

    with pytest.raises(DocumentLoadError, match="Could not open/parse"):
        load_pptx(bad_pptx)


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


def _build_zip(entries: dict) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_extract_zip_archive_extracts_supported_files(tmp_path):
    zip_bytes = _build_zip({"leave.txt": "leave content", "quality.docx": "placeholder"})

    extracted = extract_zip_archive(str(tmp_path), "Leave Policy", zip_bytes)

    assert set(extracted) == {"leave.txt", "quality.docx"}
    assert (tmp_path / "Leave Policy" / "leave.txt").read_text() == "leave content"


def test_extract_zip_archive_flattens_nested_folders_and_skips_unsupported(tmp_path):
    zip_bytes = _build_zip(
        {
            "nested/folder/leave.txt": "leave content",
            "__MACOSX/._leave.txt": "junk",
            ".DS_Store": "junk",
            "notes.unsupported": "junk",
        }
    )

    extracted = extract_zip_archive(str(tmp_path), "Leave Policy", zip_bytes)

    assert extracted == ["leave.txt"]
    assert (tmp_path / "Leave Policy" / "leave.txt").exists()


def test_extract_zip_archive_overwrites_existing_file(tmp_path):
    category_dir = tmp_path / "Leave Policy"
    category_dir.mkdir()
    (category_dir / "leave.txt").write_text("old content")

    zip_bytes = _build_zip({"leave.txt": "new content"})
    extract_zip_archive(str(tmp_path), "Leave Policy", zip_bytes)

    assert (category_dir / "leave.txt").read_text() == "new content"
