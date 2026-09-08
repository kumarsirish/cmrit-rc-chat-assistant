"""Reads docs-metadata.json: the topic -> canonical URL (and, for most
topics, local PDF filename) lookup maintained alongside policies_docs.

Used two ways:
- Ingestion tags each chunk with the topic/URL of the document it came
  from, so retrieval and the UI can surface a document's canonical
  link without re-deriving it at query time.
- Retrieval and the pipeline use find_matching_topic() to detect which
  topic (if any) a query is about, so relevant chunks can be nudged to
  the front and, for topics with no locally-indexed PDF, the URL can
  still be surfaced even though there is no chunk text to retrieve.
"""

import json
from dataclasses import dataclass

from rag.retrieval.keyword_extraction import extract_keywords


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


def find_matching_topic(query: str, topics: list):
    """Returns the DocTopic whose topic string best overlaps query's
    significant words, or None if no topic shares any.

    Scored by the size of the keyword-set intersection; ties are
    broken in favor of the topic with the higher overlap *ratio*
    (intersection size / topic's own keyword count) — i.e. the topic
    whose own defining words are more fully covered by the query, not
    the topic with the longer name. A long, multi-subject umbrella
    title (e.g. a handbook covering a dozen areas) shares generic
    words with many queries by sheer keyword count, but each such
    match covers only a small fraction of that title's own keywords;
    a short, single-subject topic name matched on the same number of
    words covers a much larger fraction of itself and is the more
    specific match.
    """
    query_keywords = set(extract_keywords(query))
    if not query_keywords:
        return None

    best_topic = None
    best_score = 0
    best_ratio = 0.0
    for doc_topic in topics:
        topic_keywords = set(extract_keywords(doc_topic.topic))
        if not topic_keywords:
            continue
        score = len(query_keywords & topic_keywords)
        if score == 0:
            continue
        ratio = score / len(topic_keywords)
        if score > best_score or (score == best_score and ratio > best_ratio):
            best_score = score
            best_ratio = ratio
            best_topic = doc_topic
    return best_topic
