"""Single source of per-document metadata.

Previously split across two files: docs-metadata.json (topic -> link /
local filename, used to detect which document a query is about) and
document_registry.json (upload bookkeeping: category, upload time,
archived/superseded tracking). Both were already keyed 1:1 by
filename, and a topic's name was always just its filename with the
extension stripped (see DocumentRecord.display_name) — so there was no
real reason to keep them as two files with two write paths that could
drift apart.

status is "active" (currently indexed) or "archived" (superseded by a
newer upload, physically moved out of the scanned directory — see
register_upload). Archived records are kept around for audit/history
(e.g. the Admin tab's "superseded by" listing) but must never be
treated as routable: active_doc_topics() is the one place that filters
on status, so a caller forgetting to filter elsewhere can't
accidentally make an archived document matchable again.

No file-locking — fine for a single-admin internal tool, same read-
modify-write pattern this replaces.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag._01_ingestion.document_loader import sanitize_topic_folder_name


@dataclass
class DocTopic:
    """One document's query-matching identity: what
    topic_matcher.EmbeddingTopicMatcher embeds and matches a query
    against, and what retrieval/response code reads link/docs_filename
    off of. Synthesized from active DocumentRecords by
    active_doc_topics() rather than stored directly, keeping the
    matcher-facing shape independent of DocumentRecord's own fields.

    docs_filename is always a list — normalizing a single filename
    string (or None) to a list here so every caller sees the same
    shape, same as the rest of this file's construction paths.

    category is the document's category/folder (see DocumentRecord) —
    carried here so retrieval_service.py can group topics by category
    and scope topic-matching to just one category's documents when the
    user has picked one (see topic_matcher.EmbeddingTopicMatcher).
    """

    topic: str
    link: str
    docs_filename: object = None
    keywords: str = None
    category: str = None

    def __post_init__(self):
        if self.docs_filename is None:
            self.docs_filename = []
        elif isinstance(self.docs_filename, str):
            self.docs_filename = [self.docs_filename]
        else:
            self.docs_filename = list(self.docs_filename)


@dataclass
class DocumentRecord:
    """One document's full bookkeeping entry: filename, category
    (folder), upload time, active/archived status and supersession
    history, plus the topic-matching fields that used to live in
    docs-metadata.json.

    link and keywords are the same canonical-URL / curated-keyword
    fields docs-metadata.json used to carry. label is an OPTIONAL
    override of display_name, meant only for documents whose raw
    filename would be a poor semantic-match target (an acronym, a
    version suffix, garbled characters from an export) — most records
    leave it unset and get their display name derived from the
    filename.
    """

    filename: str
    category: str
    upload_date: str
    status: str = "active"
    supersedes: str = None
    superseded_by: str = None
    link: str = None
    label: str = None
    keywords: str = None

    @property
    def display_name(self) -> str:
        """label if a human curated one was set, otherwise the
        filename with its extension stripped and underscores turned
        into spaces."""
        if self.label:
            return self.label
        return Path(self.filename).stem.replace("_", " ")

    @property
    def filetype(self) -> str:
        """File extension without the leading dot, lowercased —
        derived from filename rather than stored, so it can't drift
        from the actual file."""
        return Path(self.filename).suffix.lstrip(".").lower()


def load_documents(path: str) -> list:
    """Parses the documents file into a list of DocumentRecord.
    Returns an empty list if the file doesn't exist yet (first
    upload)."""
    documents_path = Path(path)
    if not documents_path.exists():
        return []
    with open(documents_path, "r", encoding="utf-8") as documents_file:
        raw = json.load(documents_file)
    return [DocumentRecord(**record) for record in raw.get("documents", [])]


def _save_documents(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as documents_file:
        json.dump({"documents": [asdict(record) for record in records]}, documents_file, indent=2, ensure_ascii=False)
        documents_file.write("\n")


def active_doc_topics(records: list) -> list:
    """Returns one DocTopic per "active" record, for
    topic_matcher.EmbeddingTopicMatcher to embed and match queries
    against. Archived (superseded) records are deliberately excluded —
    see the module docstring for why this is the one place that checks
    status."""
    return [
        DocTopic(
            topic=record.display_name,
            link=record.link,
            docs_filename=[record.filename],
            keywords=record.keywords,
            category=record.category,
        )
        for record in records
        if record.status == "active"
    ]


def active_filename_topic_map(records: list) -> dict:
    """Maps normalized filename -> DocTopic, for every active record —
    the per-chunk lookup ingestion and the Admin UI's citation links
    use to attach a topic/link to a given filename. Archived records
    are excluded, same as active_doc_topics()."""
    return {
        topic.docs_filename[0].strip().casefold(): topic
        for topic in active_doc_topics(records)
    }


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
    documents_path: str,
    source_directory: str,
    filename: str,
    category: str,
    archived_subfolder_name: str = "_archived",
    supersedes: str = None,
    link: str = None,
    label: str = None,
    keywords: str = None,
) -> DocumentRecord:
    """Records a newly uploaded document, and if supersedes (an
    existing active filename in the same category) is given, archives
    that older record: marks it "archived"/superseded_by=filename in
    this file AND physically moves its file into the category's
    archived subfolder — in that order, so a failed move leaves the
    record still saying "active" rather than claiming an archive that
    didn't happen. The old record's link/label/keywords are left as-is
    for audit purposes; it just stops being eligible for
    active_doc_topics() once its status flips.

    If no supersedes is given but an active record with this exact
    filename+category already exists, it's refreshed in place (new
    upload_date; link/label replaced and keywords appended if given)
    instead of appending a duplicate entry — this is the "overwrite"
    case (re-uploading the same filename, or a zip archive extracting
    over an existing file via extract_zip_archive): the file's content
    on disk was already overwritten by the caller before this runs, so
    this just reflects that it's a new version of the same document
    rather than a brand new one.

    Returns the (new, or refreshed existing) record for the upload.
    """
    records = load_documents(documents_path)

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
    elif existing_record := next(
        (
            record
            for record in records
            if record.filename == filename and record.category == category and record.status == "active"
        ),
        None,
    ):
        existing_record.upload_date = datetime.now(timezone.utc).isoformat()
        if link:
            existing_record.link = link
        if label:
            existing_record.label = label
        if keywords:
            existing_keywords = existing_record.keywords or ""
            existing_record.keywords = ", ".join(part for part in (existing_keywords, keywords) if part)
        _save_documents(documents_path, records)
        return existing_record

    new_record = DocumentRecord(
        filename=filename,
        category=category,
        upload_date=datetime.now(timezone.utc).isoformat(),
        supersedes=supersedes,
        link=link,
        label=label,
        keywords=keywords,
    )
    records.append(new_record)
    _save_documents(documents_path, records)
    return new_record


def list_documents_by_category(records: list) -> dict:
    """Groups records by category, for the Admin UI's listing view."""
    by_category = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)
    return by_category
