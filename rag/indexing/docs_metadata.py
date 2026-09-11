"""Reads docs-metadata.json: the topic -> canonical URL (and, for most
topics, local PDF filename) lookup maintained alongside policies_docs.

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

    docs_filename is None for entries that only point at an external
    resource (e.g. the college website, a shared Drive folder) with no
    corresponding PDF ingested into policies_docs.
    """

    topic: str
    link: str
    docs_filename: str = None


def load_doc_topics(path: str) -> list:
    """Parses docs-metadata.json into a list of DocTopic.

    Entries with no "topic" or no "link" (the section-header marker
    and the emergency-contacts entry) are skipped since they have
    nothing to match a query against or link to.
    """
    with open(path, "r", encoding="utf-8") as metadata_file:
        raw = json.load(metadata_file)

    topics = []
    for resource in raw.get("resources", []):
        topic = resource.get("topic")
        link = resource.get("link")
        if not topic or not link:
            continue
        topics.append(DocTopic(topic=topic, link=link, docs_filename=resource.get("docs")))
    return topics


def _normalize_filename(filename: str) -> str:
    return filename.strip().casefold()


def build_filename_topic_map(topics: list) -> dict:
    """Maps normalized local PDF filename -> DocTopic, for entries that
    have one (skips link-only entries)."""
    filename_map = {}
    for doc_topic in topics:
        if doc_topic.docs_filename:
            filename_map[_normalize_filename(doc_topic.docs_filename)] = doc_topic
    return filename_map
