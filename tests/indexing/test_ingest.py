import json
from types import SimpleNamespace

from rag.ingestion.chunker import Chunk
from scripts.ingest import _chunk_id, build_chunk_metadata, search_text


def _fake_config(metadata_file):
    return SimpleNamespace(documents=SimpleNamespace(metadata_file=str(metadata_file)))


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
    chunk = Chunk(
        text="Casual leave is 12 days.",
        source_file="leave.pdf",
        page_number=1,
        chunk_index=0,
        page_end=1,
        section_title="Casual Leave",
    )

    ids, metadatas = build_chunk_metadata(_fake_config(metadata_file), [chunk])

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
    }


def test_build_chunk_metadata_leaves_topic_and_link_empty_for_unmatched_file(tmp_path):
    metadata_file = _write_metadata_file(tmp_path, [])
    chunk = Chunk(text="Some text.", source_file="unlisted.pdf", page_number=1, chunk_index=0)

    _, metadatas = build_chunk_metadata(_fake_config(metadata_file), [chunk])

    assert metadatas[0]["topic"] == ""
    assert metadatas[0]["source_link"] == ""
    assert metadatas[0]["keywords"] == ""


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


def test_build_chunk_metadata_preserves_chunk_order(tmp_path):
    metadata_file = _write_metadata_file(tmp_path, [])
    chunks = [
        Chunk(text="a", source_file="a.pdf", page_number=1, chunk_index=0),
        Chunk(text="b", source_file="b.pdf", page_number=1, chunk_index=0),
    ]

    ids, metadatas = build_chunk_metadata(_fake_config(metadata_file), chunks)

    assert ids == ["a.pdf::page-1::chunk-0", "b.pdf::page-1::chunk-0"]
    assert [m["source_file"] for m in metadatas] == ["a.pdf", "b.pdf"]
