"""Upload-time bookkeeping for documents added via the Admin UI.

docs-metadata.json (see docs_metadata.py) is topic-centric — it maps a
topic to the filenames that back it, with no notion of when a file was
uploaded or whether it's still current. This module is the
upload-centric counterpart: one record per uploaded document, tracking
its category, upload time, and whether it's active or has been
superseded by a newer upload.

Supersession is manual (an admin explicitly picks which existing
document a new upload replaces — see register_upload) and enforced
physically, not just in this JSON file: the superseded file is moved
into an "_archived" subfolder that document_loader.find_document_files
excludes from its directory scan, so it stops being retrievable the
next time the index is rebuilt without needing to touch the vector
store directly. register_upload is the one place that updates this
registry and moves the file together, so the two can't drift apart.

No file-locking — fine for a single-admin internal tool, same read-
modify-write pattern as docs_metadata.add_document_to_topic.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag._01_ingestion.document_loader import sanitize_topic_folder_name


@dataclass
class DocumentRecord:
    """One uploaded document's bookkeeping entry.

    status is "active" (currently indexed) or "archived" (superseded by
    a newer upload, physically moved out of the scanned directory).
    supersedes/superseded_by hold the other record's filename, or None.
    """

    filename: str
    category: str
    upload_date: str
    status: str = "active"
    supersedes: str = None
    superseded_by: str = None


def load_registry(path: str) -> list:
    """Parses the registry file into a list of DocumentRecord. Returns
    an empty list if the file doesn't exist yet (first upload)."""
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    with open(registry_path, "r", encoding="utf-8") as registry_file:
        raw = json.load(registry_file)
    return [DocumentRecord(**record) for record in raw.get("documents", [])]


def _save_registry(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as registry_file:
        json.dump({"documents": [asdict(record) for record in records]}, registry_file, indent=2, ensure_ascii=False)
        registry_file.write("\n")


def archive_path_for(
    source_directory: str, category: str, filename: str, archived_subfolder_name: str = "_archived"
) -> Path:
    """Returns where a superseded document's file is moved to."""
    return (
        Path(source_directory)
        / sanitize_topic_folder_name(category)
        / archived_subfolder_name
        / filename
    )


def register_upload(
    registry_path: str,
    source_directory: str,
    filename: str,
    category: str,
    archived_subfolder_name: str = "_archived",
    supersedes: str = None,
) -> DocumentRecord:
    """Records a newly uploaded document, and if supersedes (an
    existing active filename in the same category) is given, archives
    that older record: marks it "archived"/superseded_by=filename in
    this registry AND physically moves its file into the category's
    archived subfolder — in that order, so a failed move leaves the
    registry still saying "active" rather than claiming an archive that
    didn't happen.

    Returns the new record for the just-registered upload.
    """
    records = load_registry(registry_path)

    if supersedes:
        old_record = next(
            (
                record
                for record in records
                if record.filename == supersedes and record.category == category and record.status == "active"
            ),
            None,
        )
        if old_record is not None:
            old_path = Path(source_directory) / sanitize_topic_folder_name(category) / old_record.filename
            new_path = archive_path_for(source_directory, category, old_record.filename, archived_subfolder_name)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))
            old_record.status = "archived"
            old_record.superseded_by = filename

    new_record = DocumentRecord(
        filename=filename,
        category=category,
        upload_date=datetime.now(timezone.utc).isoformat(),
        supersedes=supersedes,
    )
    records.append(new_record)
    _save_registry(registry_path, records)
    return new_record


def list_documents_by_category(records: list) -> dict:
    """Groups records by category, for the Admin UI's listing view."""
    by_category = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)
    return by_category
