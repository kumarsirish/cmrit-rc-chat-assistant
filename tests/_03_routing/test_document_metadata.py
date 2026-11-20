import json

from rag._03_routing.document_metadata import (
    DocumentRecord,
    active_doc_topics,
    active_filename_topic_map,
    archive_path_for,
    list_documents_by_category,
    load_documents,
    register_upload,
)


def test_load_documents_returns_empty_list_when_file_missing(tmp_path):
    assert load_documents(str(tmp_path / "missing.json")) == []


def test_display_name_derives_from_filename_when_no_label():
    record = DocumentRecord(filename="RC_Norms.docx", category="Policies", upload_date="2026-01-01")
    assert record.display_name == "RC Norms"


def test_display_name_uses_label_override_when_set():
    record = DocumentRecord(
        filename="weird-file.pdf", category="Policies", upload_date="2026-01-01", label="MEP/MEA/I&E Form"
    )
    assert record.display_name == "MEP/MEA/I&E Form"


def test_filetype_is_derived_from_filename_extension():
    record = DocumentRecord(filename="RC Norms.docx", category="Policies", upload_date="2026-01-01")
    assert record.filetype == "docx"


def test_active_doc_topics_excludes_archived_records():
    records = [
        DocumentRecord(
            filename="leave-v1.pdf",
            category="Leave Policy",
            upload_date="2026-01-01",
            status="archived",
            superseded_by="leave-v2.pdf",
            link="https://example.com/leave",
        ),
        DocumentRecord(
            filename="leave-v2.pdf",
            category="Leave Policy",
            upload_date="2026-01-02",
            link="https://example.com/leave",
        ),
    ]

    topics = active_doc_topics(records)

    assert len(topics) == 1
    assert topics[0].docs_filename == ["leave-v2.pdf"]
    assert topics[0].topic == "leave-v2"
    assert topics[0].category == "Leave Policy"


def test_active_filename_topic_map_is_keyed_by_normalized_filename():
    records = [
        DocumentRecord(filename="Leave.pdf", category="Leave Policy", upload_date="2026-01-01"),
    ]
    filename_map = active_filename_topic_map(records)
    assert filename_map["leave.pdf"].docs_filename == ["Leave.pdf"]


def test_register_upload_appends_a_new_active_record(tmp_path):
    documents_path = tmp_path / "documents.json"

    record = register_upload(str(documents_path), str(tmp_path), "leave.pdf", "Leave Policy")

    assert record.filename == "leave.pdf"
    assert record.category == "Leave Policy"
    assert record.status == "active"
    assert record.supersedes is None

    records = load_documents(str(documents_path))
    assert len(records) == 1
    assert records[0].filename == "leave.pdf"


def test_register_upload_stores_link_and_label(tmp_path):
    documents_path = tmp_path / "documents.json"

    register_upload(
        str(documents_path),
        str(tmp_path),
        "leave.pdf",
        "Leave Policy",
        link="https://example.com/leave",
        label="Leave Information",
    )

    records = load_documents(str(documents_path))
    assert records[0].link == "https://example.com/leave"
    assert records[0].display_name == "Leave Information"


def test_register_upload_with_supersedes_archives_the_old_record_and_moves_its_file(tmp_path):
    documents_path = tmp_path / "documents.json"
    category_dir = tmp_path / "Leave Policy"
    category_dir.mkdir()
    old_file = category_dir / "leave-v1.pdf"
    old_file.write_text("old content")

    register_upload(str(documents_path), str(tmp_path), "leave-v1.pdf", "Leave Policy")
    register_upload(
        str(documents_path), str(tmp_path), "leave-v2.pdf", "Leave Policy", supersedes="leave-v1.pdf"
    )

    records = load_documents(str(documents_path))
    old_record = next(r for r in records if r.filename == "leave-v1.pdf")
    new_record = next(r for r in records if r.filename == "leave-v2.pdf")

    assert old_record.status == "archived"
    assert old_record.superseded_by == "leave-v2.pdf"
    assert new_record.supersedes == "leave-v1.pdf"

    assert not old_file.exists()
    archived_path = archive_path_for(str(tmp_path), "Leave Policy", "leave-v1.pdf")
    assert archived_path.exists()
    assert archived_path.read_text() == "old content"

    # The archived record must never show up as routable again.
    active_filenames = [topic.docs_filename[0] for topic in active_doc_topics(records)]
    assert "leave-v1.pdf" not in active_filenames
    assert "leave-v2.pdf" in active_filenames


def test_register_upload_with_same_filename_refreshes_upload_date_instead_of_duplicating(tmp_path):
    documents_path = tmp_path / "documents.json"

    first = register_upload(str(documents_path), str(tmp_path), "leave.pdf", "Leave Policy")
    second = register_upload(str(documents_path), str(tmp_path), "leave.pdf", "Leave Policy")

    records = load_documents(str(documents_path))
    assert len(records) == 1
    assert second.upload_date >= first.upload_date


def test_register_upload_with_supersedes_but_no_matching_active_record_is_a_noop_archive(tmp_path):
    documents_path = tmp_path / "documents.json"

    record = register_upload(
        str(documents_path), str(tmp_path), "leave-v2.pdf", "Leave Policy", supersedes="nonexistent.pdf"
    )

    assert record.filename == "leave-v2.pdf"
    records = load_documents(str(documents_path))
    assert len(records) == 1


def test_list_documents_by_category_groups_records(tmp_path):
    documents_path = tmp_path / "documents.json"
    register_upload(str(documents_path), str(tmp_path), "leave.pdf", "Leave Policy")
    register_upload(str(documents_path), str(tmp_path), "quality.pdf", "Quality")

    by_category = list_documents_by_category(load_documents(str(documents_path)))

    assert set(by_category.keys()) == {"Leave Policy", "Quality"}
    assert by_category["Leave Policy"][0].filename == "leave.pdf"


def test_archive_path_for_uses_sanitized_category_and_archived_subfolder(tmp_path):
    path = archive_path_for(str(tmp_path), "Leave / Policy", "leave.pdf", archived_subfolder_name="_archived")
    assert path == tmp_path / "Leave - Policy" / "_archived" / "leave.pdf"


def test_load_documents_parses_file_contents(tmp_path):
    documents_path = tmp_path / "documents.json"
    documents_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "filename": "leave.pdf",
                        "category": "Leave Policy",
                        "upload_date": "2026-01-01T00:00:00+00:00",
                        "link": "https://example.com/leave",
                    }
                ]
            }
        )
    )

    records = load_documents(str(documents_path))
    assert len(records) == 1
    assert records[0].filename == "leave.pdf"
    assert records[0].link == "https://example.com/leave"
