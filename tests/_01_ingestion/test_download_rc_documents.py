from helper_scripts.download_rc_documents import (
    DownloadLink,
    _extension_from_response,
    _sanitize_filename_stem,
    classify_link,
    find_download_links,
)


def test_classify_link_resolves_google_doc_to_docx_export():
    link = classify_link(
        "https://docs.google.com/document/d/ABC123/edit?usp=sharing", "RC Norms"
    )
    assert link.extension == ".docx"
    assert link.export_url == "https://docs.google.com/document/d/ABC123/export?format=docx"
    assert link.label == "RC Norms"
    assert link.source_url == "https://docs.google.com/document/d/ABC123/edit?usp=sharing"


def test_classify_link_resolves_google_sheet_to_xlsx_export():
    link = classify_link("https://docs.google.com/spreadsheets/d/XYZ789/edit", "Skill Matrix")
    assert link.extension == ".xlsx"
    assert link.export_url == "https://docs.google.com/spreadsheets/d/XYZ789/export?format=xlsx"


def test_classify_link_resolves_google_slides_to_pptx_export():
    link = classify_link("https://docs.google.com/presentation/d/SLD001/edit", "Mega Project")
    assert link.extension == ".pptx"
    assert link.export_url == "https://docs.google.com/presentation/d/SLD001/export/pptx"


def test_classify_link_resolves_drive_file_link_with_unknown_extension():
    link = classify_link("https://drive.google.com/file/d/FILE001/view?usp=sharing", "RCPIE Targets")
    assert link.extension is None
    assert link.export_url == "https://drive.google.com/uc?export=download&id=FILE001"


def test_classify_link_resolves_legacy_open_id_link():
    link = classify_link("https://drive.google.com/open?id=OPEN001", "COE Mega Applications")
    assert link.export_url == "https://drive.google.com/uc?export=download&id=OPEN001"


def test_classify_link_returns_none_for_unrelated_url():
    assert classify_link("https://cmrit.ac.in/about", "About us") is None


def test_sanitize_filename_stem_strips_unsafe_characters():
    assert _sanitize_filename_stem("MEP/MEA/I&E Project Proposal Form") == "MEPMEAI&E Project Proposal Form"


def test_sanitize_filename_stem_falls_back_when_nothing_usable_is_left():
    assert _sanitize_filename_stem("///") == "document"


def test_extension_from_response_reads_content_disposition_filename():
    class FakeResponse:
        headers = {"Content-Disposition": 'attachment; filename="RCPIE_Targets.pdf"'}

    assert _extension_from_response(FakeResponse()) == ".pdf"


def test_extension_from_response_returns_none_without_a_filename():
    class FakeResponse:
        headers = {}

    assert _extension_from_response(FakeResponse()) is None


class _FakeHtmlSession:
    def __init__(self, html):
        self._html = html

    def get(self, url, headers=None, timeout=None):
        return _FakeHtmlResponse(self._html)


class _FakeHtmlResponse:
    def __init__(self, html):
        self.text = html

    def raise_for_status(self):
        pass


def test_find_download_links_extracts_distinct_recognized_links():
    html = """
    <html><body>
        <a href="https://docs.google.com/document/d/DOC1/edit">RC Norms</a>
        <a href="https://drive.google.com/open?id=DRIVE1">RCPIE Targets</a>
        <a href="https://cmrit.ac.in/about">About (should be ignored)</a>
        <a href="https://docs.google.com/document/d/DOC1/edit?ouid=123">RC Norms (duplicate link)</a>
    </body></html>
    """
    links = find_download_links("https://example.com/downloads", _FakeHtmlSession(html))

    assert [link.label for link in links] == ["RC Norms", "RCPIE Targets"]
    assert links[0].export_url == "https://docs.google.com/document/d/DOC1/export?format=docx"
