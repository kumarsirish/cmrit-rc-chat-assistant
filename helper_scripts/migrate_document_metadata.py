"""One-time migration: merges config/docs-metadata.json (topic/link
lookup) and config/document_registry.json (upload bookkeeping) into
the single config/documents.json consumed by
rag._03_routing.document_metadata.

Joins on filename — the natural key both old files agreed on for
every entry except one ("Top MSME Funding.docx" exists only in the
registry, with no docs-metadata.json counterpart; it migrates with
link/keywords left unset). The standalone "topic" field is dropped: in
every old docs-metadata.json entry it was just the filename with the
extension stripped (two entries differ only by filesystem-unsafe
characters that were stripped from the filename but kept in the topic
string, which display_name's own stripping doesn't replicate, so those
two migrate with an explicit label override to preserve their exact
former topic text).

Run once:

    python helper_scripts/migrate_document_metadata.py

Safe to re-run: it only reads the two old files and overwrites the
merged output, never the old files themselves.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag._03_routing.document_metadata import DocumentRecord, _save_documents

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_METADATA_PATH = _REPO_ROOT / "config" / "docs-metadata.json"
_DEFAULT_REGISTRY_PATH = _REPO_ROOT / "config" / "document_registry.json"
_DEFAULT_DOCUMENTS_PATH = _REPO_ROOT / "config" / "documents.json"


def _load_topic_lookup(metadata_path: str) -> dict:
    """filename (as it literally appears in docs-metadata.json's
    "docs") -> {"topic": ..., "link": ..., "keywords": ...}."""
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        raw = json.load(metadata_file)

    lookup = {}
    for resource in raw.get("resources", []):
        docs = resource.get("docs")
        if docs is None:
            continue
        docs = [docs] if isinstance(docs, str) else docs
        for filename in docs:
            lookup[filename] = {
                "topic": resource.get("topic"),
                "link": resource.get("link") or None,
                "keywords": resource.get("keywords") or None,
            }
    return lookup


def _label_for(filename: str, topic: str) -> str:
    """The old topic string is only worth keeping as an explicit label
    when it disagrees with what DocumentRecord.display_name would
    derive from the filename on its own (i.e. a document whose
    filename had filesystem-unsafe characters stripped that the topic
    string kept) — every other entry migrates with no label, so its
    display name is derived from the filename going forward."""
    derived = Path(filename).stem.replace("_", " ")
    return topic if topic and topic != derived else None


def migrate(metadata_path: str, registry_path: str, documents_path: str) -> list:
    topic_lookup = _load_topic_lookup(metadata_path)

    with open(registry_path, "r", encoding="utf-8") as registry_file:
        registry_raw = json.load(registry_file)

    merged_records = []
    seen_filenames = set()
    for entry in registry_raw.get("documents", []):
        filename = entry["filename"]
        seen_filenames.add(filename)
        topic_info = topic_lookup.get(filename, {})
        merged_records.append(
            DocumentRecord(
                filename=filename,
                category=entry["category"],
                upload_date=entry["upload_date"],
                status=entry.get("status", "active"),
                supersedes=entry.get("supersedes"),
                superseded_by=entry.get("superseded_by"),
                link=topic_info.get("link"),
                label=_label_for(filename, topic_info.get("topic")),
                keywords=topic_info.get("keywords"),
            )
        )

    # Any docs-metadata.json filename the registry doesn't know about
    # (none exist today, but a future drift shouldn't silently drop
    # data) still gets a record, with no upload/category bookkeeping.
    for filename, topic_info in topic_lookup.items():
        if filename in seen_filenames:
            continue
        merged_records.append(
            DocumentRecord(
                filename=filename,
                category="(uncategorized)",
                upload_date=None,
                link=topic_info.get("link"),
                label=_label_for(filename, topic_info.get("topic")),
                keywords=topic_info.get("keywords"),
            )
        )

    _save_documents(documents_path, merged_records)
    return merged_records


if __name__ == "__main__":
    records = migrate(str(_DEFAULT_METADATA_PATH), str(_DEFAULT_REGISTRY_PATH), str(_DEFAULT_DOCUMENTS_PATH))
    print(f"Wrote {len(records)} document record(s) to {_DEFAULT_DOCUMENTS_PATH}")
