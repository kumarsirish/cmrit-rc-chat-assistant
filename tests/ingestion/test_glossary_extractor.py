import json

from rag.ingestion.document_loader import PageDocument
from rag.ingestion.glossary_extractor import extract_glossary, load_glossary, save_glossary


def _page(text: str) -> PageDocument:
    return PageDocument(text=text, source_file="test.pdf", page_number=1)


def test_extract_glossary_finds_phrase_then_acronym():
    pages = [_page("The Internal Quality Assurance Cell (IQAC) audits each department twice a semester.")]
    glossary = extract_glossary(pages)
    assert glossary["IQAC"] == "Internal Quality Assurance Cell"


def test_extract_glossary_finds_acronym_then_phrase():
    pages = [_page("Contact the DRC (Department Research Committee) for approval.")]
    glossary = extract_glossary(pages)
    assert glossary["DRC"] == "Department Research Committee"


def test_extract_glossary_trims_leading_noise_words():
    pages = [
        _page(
            "the faculty at CMRIT pursue research, consultancy, projects, innovation "
            "and entrepreneurship (RCPIE)."
        )
    ]
    glossary = extract_glossary(pages)
    assert glossary["RCPIE"] == "research consultancy projects innovation and entrepreneurship"


def test_extract_glossary_rejects_non_matching_initials():
    pages = [_page("This document was prepared by the committee (ABC) last year.")]
    glossary = extract_glossary(pages)
    assert "ABC" not in glossary


def test_extract_glossary_drops_acronyms_shorter_than_min_length():
    pages = [_page("Employees may take Casual Leave (CL) or Vacation Leave (VL).")]
    glossary = extract_glossary(pages, min_acronym_length=3)
    assert "CL" not in glossary
    assert "VL" not in glossary


def test_extract_glossary_prefers_most_common_and_non_upper_phrasing():
    pages = [
        _page("Peer Support Team (PST) members help classmates."),
        _page("Peer Support Team (PST) members report to class teachers."),
        _page("PEER SUPPORT TEAM (PST) — annual training."),
    ]
    glossary = extract_glossary(pages)
    assert glossary["PST"] == "Peer Support Team"


def test_save_and_load_glossary_round_trip(tmp_path):
    glossary = {"IQAC": "Internal Quality Assurance Cell", "RCPIE": "Research Consultancy"}
    path = tmp_path / "glossary.json"
    save_glossary(glossary, str(path))

    loaded = load_glossary(str(path))
    assert loaded == glossary
    with open(path) as glossary_file:
        assert json.load(glossary_file) == dict(sorted(glossary.items()))


def test_load_glossary_returns_empty_dict_when_missing(tmp_path):
    assert load_glossary(str(tmp_path / "does-not-exist.json")) == {}
