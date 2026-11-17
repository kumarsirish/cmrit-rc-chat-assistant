import json
from types import SimpleNamespace

from rag._01_ingestion.chunker import Chunk
from helper_scripts.ingest import (
    UNCATEGORIZED,
    _build_filename_location_map,
    _chunk_id,
    build_chunk_metadata,
    dedupe_chunks,
    search_text,
)


def _fake_config(metadata_file, source_directory=None, archived_subfolder_name="_archived"):
    return SimpleNamespace(
        documents=SimpleNamespace(
            metadata_file=str(metadata_file),
            # Nonexistent by default — _build_filename_location_map
            # tolerates that (returns {}), which is fine for tests that
            # don't care about category/source_path specifics.
            source_directory=str(source_directory) if source_directory else "/nonexistent-for-tests",
            archived_subfolder_name=archived_subfolder_name,
        )
    )


def _write_metadata_file(tmp_path, resources):
    path = tmp_path / "docs-metadata.json"
    path.write_text(json.dumps({"resources": resources}), encoding="utf-8")
    return path


def test_chunk_id_is_stable_and_unique_per_chunk():
    chunk = Chunk(text="x", source_file="leave.pdf", page_number=2, chunk_index=3)
    assert _chunk_id(chunk) == "leave.pdf::page-2::chunk-3"


def test_build_chunk_metadata_matches_chunk_to_its_topic_by_filename(tmp_path):
    metadata_file = _write_metadata_file(
        tmp_path,
        [{"topic": "Leave Policy", "link": "https://example.com/leave", "docs": "leave.pdf"}],
    )
    source_directory = tmp_path / "knowledge_documents"
    category_dir = source_directory / "Leave Policy"
    category_dir.mkdir(parents=True)
    (category_dir / "leave.pdf").write_bytes(b"x")

    chunk = Chunk(
        text="Casual leave is 12 days.",
        source_file="leave.pdf",
        page_number=1,
        chunk_index=0,
        page_end=1,
        section_title="Casual Leave",
    )

    ids, metadatas = build_chunk_metadata(_fake_config(metadata_file, source_directory), [chunk])

    assert ids == ["leave.pdf::page-1::chunk-0"]
    assert metadatas[0] == {
        "source_file": "leave.pdf",
        "page_number": 1,
        "page_end": 1,
        "section_title": "Casual Leave",
        "chunk_index": 0,
        "topic": "Leave Policy",
        "source_link": "https://example.com/leave",
        "keywords": "",
        "category": "Leave Policy",
        "source": "leave.pdf",
        "source_path": "Leave Policy/leave.pdf",
        "page": 1,
        "chunk_id": "leave.pdf::page-1::chunk-0",
        "document_type": "pdf",
    }


def test_build_chunk_metadata_uses_uncategorized_for_a_file_with_no_category_subfolder(tmp_path):
    metadata_file = _write_metadata_file(tmp_path, [])
    source_directory = tmp_path / "knowledge_documents"
    source_directory.mkdir()
    (source_directory / "unlisted.pdf").write_bytes(b"x")

    chunk = Chunk(text="Some text.", source_file="unlisted.pdf", page_number=1, chunk_index=0)

    _, metadatas = build_chunk_metadata(_fake_config(metadata_file, source_directory), [chunk])

    assert metadatas[0]["topic"] == ""
    assert metadatas[0]["source_link"] == ""
    assert metadatas[0]["keywords"] == ""
    assert metadatas[0]["category"] == UNCATEGORIZED
    assert metadatas[0]["source_path"] == "unlisted.pdf"
    assert metadatas[0]["document_type"] == "pdf"


def test_build_chunk_metadata_falls_back_to_uncategorized_when_source_directory_is_missing(tmp_path):
    metadata_file = _write_metadata_file(tmp_path, [])
    chunk = Chunk(text="Some text.", source_file="unlisted.pdf", page_number=1, chunk_index=0)

    _, metadatas = build_chunk_metadata(_fake_config(metadata_file), [chunk])

    assert metadatas[0]["category"] == UNCATEGORIZED
    assert metadatas[0]["source_path"] == "unlisted.pdf"


def test_build_chunk_metadata_document_type_matches_extension(tmp_path):
    metadata_file = _write_metadata_file(tmp_path, [])
    chunk = Chunk(text="Slide text.", source_file="deck.pptx", page_number=1, chunk_index=0)

    _, metadatas = build_chunk_metadata(_fake_config(metadata_file), [chunk])

    assert metadatas[0]["document_type"] == "pptx"


def test_build_filename_location_map_returns_empty_dict_when_directory_missing(tmp_path):
    assert _build_filename_location_map(str(tmp_path / "missing"), "_archived") == {}


def test_build_filename_location_map_excludes_archived_files(tmp_path):
    category_dir = tmp_path / "Writing guides"
    category_dir.mkdir()
    (category_dir / "active.docx").write_bytes(b"x")
    archived_dir = category_dir / "_archived"
    archived_dir.mkdir()
    (archived_dir / "old.docx").write_bytes(b"x")

    location_map = _build_filename_location_map(str(tmp_path), "_archived")

    assert "active.docx" in location_map
    assert "old.docx" not in location_map


def test_build_chunk_metadata_carries_topic_keywords(tmp_path):
    metadata_file = _write_metadata_file(
        tmp_path,
        [
            {
                "topic": "Roles and Responsibilities",
                "link": "https://example.com/roles",
                "docs": "roles.pdf",
                "keywords": "Director, Principal, HOD",
            }
        ],
    )
    chunk = Chunk(text="1. Leading functions and operations.", source_file="roles.pdf", page_number=1, chunk_index=0)

    _, metadatas = build_chunk_metadata(_fake_config(metadata_file), [chunk])

    assert metadatas[0]["keywords"] == "Director, Principal, HOD"


def test_search_text_appends_keywords_without_changing_chunk_text():
    text = "1. Leading functions and operations."
    assert search_text(text, "Director, Principal, HOD") == (
        "1. Leading functions and operations.\n\nKeywords: Director, Principal, HOD"
    )


def test_search_text_is_unchanged_when_no_keywords():
    text = "1. Leading functions and operations."
    assert search_text(text, "") == text


def test_dedupe_chunks_drops_identical_text_from_a_different_document():
    chunks = [
        Chunk(text="About CMRIT: established in 2000.", source_file="faculty.pdf", page_number=1, chunk_index=0),
        Chunk(text="About CMRIT: established in 2000.", source_file="student.pdf", page_number=1, chunk_index=0),
    ]
    deduped = dedupe_chunks(chunks)
    assert len(deduped) == 1
    assert deduped[0].source_file == "faculty.pdf"


def test_dedupe_chunks_ignores_whitespace_differences():
    chunks = [
        Chunk(text="About  CMRIT:\nestablished in 2000.", source_file="faculty.pdf", page_number=1, chunk_index=0),
        Chunk(text="About CMRIT: established   in 2000.", source_file="student.pdf", page_number=1, chunk_index=0),
    ]
    assert len(dedupe_chunks(chunks)) == 1


def test_dedupe_chunks_keeps_distinct_content():
    chunks = [
        Chunk(text="Casual leave is 12 days.", source_file="leave.pdf", page_number=1, chunk_index=0),
        Chunk(text="Vacation leave is 7 days.", source_file="leave.pdf", page_number=2, chunk_index=0),
    ]
    deduped = dedupe_chunks(chunks)
    assert len(deduped) == 2


def test_dedupe_chunks_preserves_order_of_first_occurrences():
    chunks = [
        Chunk(text="a", source_file="a.pdf", page_number=1, chunk_index=0),
        Chunk(text="b", source_file="b.pdf", page_number=1, chunk_index=0),
        Chunk(text="a", source_file="c.pdf", page_number=1, chunk_index=0),
    ]
    deduped = dedupe_chunks(chunks)
    assert [c.source_file for c in deduped] == ["a.pdf", "b.pdf"]


def test_build_chunk_metadata_preserves_chunk_order(tmp_path):
    metadata_file = _write_metadata_file(tmp_path, [])
    chunks = [
        Chunk(text="a", source_file="a.pdf", page_number=1, chunk_index=0),
        Chunk(text="b", source_file="b.pdf", page_number=1, chunk_index=0),
    ]

    ids, metadatas = build_chunk_metadata(_fake_config(metadata_file), chunks)

    assert ids == ["a.pdf::page-1::chunk-0", "b.pdf::page-1::chunk-0"]
    assert [m["source_file"] for m in metadatas] == ["a.pdf", "b.pdf"]
