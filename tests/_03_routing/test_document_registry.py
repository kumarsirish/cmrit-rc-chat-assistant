from rag._03_routing.document_registry import (
    archive_path_for,
    list_documents_by_category,
    load_registry,
    register_upload,
)


def test_load_registry_returns_empty_list_when_file_missing(tmp_path):
    assert load_registry(str(tmp_path / "missing.json")) == []


def test_register_upload_appends_a_new_active_record(tmp_path):
    registry_path = tmp_path / "registry.json"

    record = register_upload(str(registry_path), str(tmp_path), "leave.pdf", "Leave Policy")

    assert record.filename == "leave.pdf"
    assert record.category == "Leave Policy"
    assert record.status == "active"
    assert record.supersedes is None

    records = load_registry(str(registry_path))
    assert len(records) == 1
    assert records[0].filename == "leave.pdf"


def test_register_upload_with_supersedes_archives_the_old_record_and_moves_its_file(tmp_path):
    registry_path = tmp_path / "registry.json"
    category_dir = tmp_path / "Leave Policy"
    category_dir.mkdir()
    old_file = category_dir / "leave-v1.pdf"
    old_file.write_text("old content")

    register_upload(str(registry_path), str(tmp_path), "leave-v1.pdf", "Leave Policy")
    register_upload(
        str(registry_path), str(tmp_path), "leave-v2.pdf", "Leave Policy", supersedes="leave-v1.pdf"
    )

    records = load_registry(str(registry_path))
    old_record = next(r for r in records if r.filename == "leave-v1.pdf")
    new_record = next(r for r in records if r.filename == "leave-v2.pdf")

    assert old_record.status == "archived"
    assert old_record.superseded_by == "leave-v2.pdf"
    assert new_record.supersedes == "leave-v1.pdf"

    assert not old_file.exists()
    archived_path = archive_path_for(str(tmp_path), "Leave Policy", "leave-v1.pdf")
    assert archived_path.exists()
    assert archived_path.read_text() == "old content"


def test_register_upload_with_same_filename_refreshes_upload_date_instead_of_duplicating(tmp_path):
    registry_path = tmp_path / "registry.json"

    first = register_upload(str(registry_path), str(tmp_path), "leave.pdf", "Leave Policy")
    second = register_upload(str(registry_path), str(tmp_path), "leave.pdf", "Leave Policy")

    records = load_registry(str(registry_path))
    assert len(records) == 1
    assert second.upload_date >= first.upload_date


def test_register_upload_with_supersedes_but_no_matching_active_record_is_a_noop_archive(tmp_path):
    registry_path = tmp_path / "registry.json"

    record = register_upload(
        str(registry_path), str(tmp_path), "leave-v2.pdf", "Leave Policy", supersedes="nonexistent.pdf"
    )

    assert record.filename == "leave-v2.pdf"
    records = load_registry(str(registry_path))
    assert len(records) == 1


def test_list_documents_by_category_groups_records(tmp_path):
    registry_path = tmp_path / "registry.json"
    register_upload(str(registry_path), str(tmp_path), "leave.pdf", "Leave Policy")
    register_upload(str(registry_path), str(tmp_path), "quality.pdf", "Quality")

    by_category = list_documents_by_category(load_registry(str(registry_path)))

    assert set(by_category.keys()) == {"Leave Policy", "Quality"}
    assert by_category["Leave Policy"][0].filename == "leave.pdf"


def test_archive_path_for_uses_sanitized_category_and_archived_subfolder(tmp_path):
    path = archive_path_for(str(tmp_path), "Leave / Policy", "leave.pdf", archived_subfolder_name="_archived")
    assert path == tmp_path / "Leave - Policy" / "_archived" / "leave.pdf"
