"""Reads and writes docs-metadata.json: the topic -> canonical URL
(and, for most topics, local PDF filename) lookup maintained alongside
the documents source directory.

Used two ways:
- Ingestion tags each chunk with the topic/URL of the document it came
  from, so retrieval and the UI can surface a document's canonical
  link without re-deriving it at query time.
- Retrieval and the pipeline use topic_matcher.EmbeddingTopicMatcher
  (built from the DocTopic list this module produces) to detect which
  topic (if any) a query is about, so relevant chunks can be nudged to
  the front and, for topics with no locally-indexed PDF, the URL can
  still be surfaced even though there is no chunk text to retrieve.
"""

import json
from dataclasses import dataclass


@dataclass
class DocTopic:
    """One topic -> URL mapping from docs-metadata.json.

    docs_filename is always a list — empty for entries that only point
    at an external resource (e.g. the college website, a shared Drive
    folder) with no corresponding PDF ingested into knowledge_documents, and
    can hold more than one filename for a topic backed by several
    local PDFs. Accepts a single filename string, a list, or None at
    construction time (matching what "docs" in docs-metadata.json can
    be) and normalizes to a list here so every caller — whether built
    via load_doc_topics or constructed directly, e.g. in tests — sees
    the same shape.

    keywords holds extra terms (job titles, synonyms, etc.) that a
    query might use but the topic name itself doesn't contain — e.g.
    "Roles and Responsibilities" says nothing about "Director" or
    "duties". topic_matcher folds these into the text it embeds for
    this topic, so short/generic topic names still match the queries
    they should.
    """

    topic: str
    link: str
    docs_filename: object = None
    keywords: str = None

    def __post_init__(self):
        if self.docs_filename is None:
            self.docs_filename = []
        elif isinstance(self.docs_filename, str):
            self.docs_filename = [self.docs_filename]
        else:
            self.docs_filename = list(self.docs_filename)


def load_doc_topics(path: str) -> list:
    """Parses docs-metadata.json into a list of DocTopic.

    Entries with no "topic", or with neither a "link" nor a "docs"
    (the emergency-contacts entry), are skipped since they have
    nothing to match a query against or link to. A topic can have an
    empty "link" (e.g. pending a canonical URL) as long as it still
    has a local "docs" PDF to match and retrieve from.
    """
    with open(path, "r", encoding="utf-8") as metadata_file:
        raw = json.load(metadata_file)

    topics = []
    for resource in raw.get("resources", []):
        topic = resource.get("topic")
        link = resource.get("link")
        docs = resource.get("docs")
        if not topic or not (link or docs):
            continue
        topics.append(
            DocTopic(
                topic=topic,
                link=link,
                docs_filename=resource.get("docs"),
                keywords=resource.get("keywords"),
            )
        )
    return topics


def add_document_to_topic(path: str, topic: str, filename: str, link: str = None, keywords: str = None) -> None:
    """Adds filename to topic's "docs" list in docs-metadata.json,
    creating the topic entry if none matches yet (matched
    case-insensitively against an existing topic name so re-uploading
    under slightly different casing doesn't create a duplicate entry).

    link, if given, replaces the entry's link. keywords, if given, is
    appended to any keywords already there rather than replacing them,
    since a topic can accumulate keywords across several uploads.

    Used by the Admin UI's upload flow so a newly uploaded document is
    immediately retrievable under the right topic without hand-editing
    this file.
    """
    with open(path, "r", encoding="utf-8") as metadata_file:
        raw = json.load(metadata_file)

    resources = raw.setdefault("resources", [])
    normalized_topic = topic.strip().casefold()
    resource = next(
        (entry for entry in resources if (entry.get("topic") or "").strip().casefold() == normalized_topic),
        None,
    )
    if resource is None:
        resource = {"topic": topic.strip()}
        resources.append(resource)

    docs = resource.get("docs")
    if docs is None:
        docs = []
    elif isinstance(docs, str):
        docs = [docs]
    else:
        docs = list(docs)
    if filename not in docs:
        docs.append(filename)
    resource["docs"] = docs

    if link:
        resource["link"] = link
    if keywords:
        existing_keywords = resource.get("keywords") or ""
        resource["keywords"] = ", ".join(part for part in (existing_keywords, keywords) if part)

    with open(path, "w", encoding="utf-8") as metadata_file:
        json.dump(raw, metadata_file, indent=2, ensure_ascii=False)
        metadata_file.write("\n")


def remove_document_from_topic(path: str, topic: str, filename: str) -> None:
    """Removes filename from topic's "docs" list in docs-metadata.json
    (matched case-insensitively against the topic name, same as
    add_document_to_topic). No-op if the topic or filename isn't found.

    Used when a document is superseded (see
    rag._03_routing.document_registry) so the old filename doesn't sit
    around as dead metadata pointing at a file that's no longer
    ingested.
    """
    with open(path, "r", encoding="utf-8") as metadata_file:
        raw = json.load(metadata_file)

    resources = raw.get("resources", [])
    normalized_topic = topic.strip().casefold()
    resource = next(
        (entry for entry in resources if (entry.get("topic") or "").strip().casefold() == normalized_topic),
        None,
    )
    if resource is None:
        return

    docs = resource.get("docs")
    if docs is None:
        return
    docs = [docs] if isinstance(docs, str) else list(docs)
    if filename not in docs:
        return
    docs.remove(filename)
    resource["docs"] = docs

    with open(path, "w", encoding="utf-8") as metadata_file:
        json.dump(raw, metadata_file, indent=2, ensure_ascii=False)
        metadata_file.write("\n")


def _normalize_filename(filename: str) -> str:
    return filename.strip().casefold()


def build_filename_topic_map(topics: list) -> dict:
    """Maps normalized local PDF filename -> DocTopic, for every
    filename an entry has (skips link-only entries). A topic backed by
    several PDFs (docs_filename with more than one entry) gets a
    filename_map entry for each of them, all pointing at the same
    DocTopic."""
    filename_map = {}
    for doc_topic in topics:
        for filename in doc_topic.docs_filename:
            filename_map[_normalize_filename(filename)] = doc_topic
    return filename_map
