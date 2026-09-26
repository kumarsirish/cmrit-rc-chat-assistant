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

    docs_filename is always a list — empty for entries that only point
    at an external resource (e.g. the college website, a shared Drive
    folder) with no corresponding PDF ingested into policies_docs, and
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
        topics.append(
            DocTopic(
                topic=topic,
                link=link,
                docs_filename=resource.get("docs"),
                keywords=resource.get("keywords"),
            )
        )
    return topics


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
