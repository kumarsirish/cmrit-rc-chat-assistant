import json

from rag._03_routing.docs_metadata import (
    DocTopic,
    add_document_to_topic,
    build_filename_topic_map,
    load_doc_topics,
    remove_document_from_topic,
)


def _sample_topics():
    return [
        DocTopic(
            topic="Leave Information, Application and Approval Process",
            link="https://example.com/leave",
            docs_filename="CMRIT - Leave Information.pdf",
        ),
        DocTopic(
            topic="CMRIT Website",
            link="http://www.cmrit.ac.in",
            docs_filename=None,
        ),
        DocTopic(
            topic="Student Handbook",
            link="https://example.com/student-handbook",
            docs_filename="Student Handbook-CMRIT.docx.pdf",
        ),
    ]


def test_build_filename_topic_map_normalizes_and_skips_link_only():
    topics = _sample_topics()
    filename_map = build_filename_topic_map(topics)
    assert len(filename_map) == 2
    assert filename_map["cmrit - leave information.pdf"].topic == (
        "Leave Information, Application and Approval Process"
    )
    assert "cmrit website" not in filename_map


def test_build_filename_topic_map_maps_every_file_of_a_multi_doc_topic():
    topic = DocTopic(
        topic="Leave Policy",
        link="https://example.com/leave",
        docs_filename=["leave.pdf", "leave-addendum.pdf"],
    )
    filename_map = build_filename_topic_map([topic])
    assert filename_map["leave.pdf"] is topic
    assert filename_map["leave-addendum.pdf"] is topic


def test_load_doc_topics_keeps_entries_with_docs_but_no_link(tmp_path):
    metadata_file = tmp_path / "docs-metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "topic": "Probation Period Completion Process",
                        "link": "",
                        "docs": "Probation.pdf",
                    },
                    {
                        "topic": "For Emergency related SPOC",
                        "contacts": {"medical_emergency": {"phone": "123"}},
                    },
                    {"topic": None, "link": "", "docs": "orphan.pdf"},
                ]
            }
        )
    )

    topics = load_doc_topics(str(metadata_file))

    assert [t.topic for t in topics] == ["Probation Period Completion Process"]
    assert topics[0].link == ""
    assert topics[0].docs_filename == ["Probation.pdf"]


def _write_metadata(path, resources):
    path.write_text(json.dumps({"resources": resources}))


def test_add_document_to_topic_creates_new_topic_entry(tmp_path):
    metadata_file = tmp_path / "docs-metadata.json"
    _write_metadata(metadata_file, [])

    add_document_to_topic(str(metadata_file), "Leave Policy", "leave.pdf", link="https://example.com/leave")

    topics = load_doc_topics(str(metadata_file))
    assert len(topics) == 1
    assert topics[0].topic == "Leave Policy"
    assert topics[0].docs_filename == ["leave.pdf"]
    assert topics[0].link == "https://example.com/leave"


def test_add_document_to_topic_appends_to_existing_topic_case_insensitively(tmp_path):
    metadata_file = tmp_path / "docs-metadata.json"
    _write_metadata(metadata_file, [{"topic": "leave policy", "link": "", "docs": ["leave-v1.pdf"]}])

    add_document_to_topic(str(metadata_file), "Leave Policy", "leave-v2.pdf")

    topics = load_doc_topics(str(metadata_file))
    assert len(topics) == 1
    assert topics[0].docs_filename == ["leave-v1.pdf", "leave-v2.pdf"]


def test_remove_document_from_topic_drops_the_filename(tmp_path):
    metadata_file = tmp_path / "docs-metadata.json"
    _write_metadata(
        metadata_file, [{"topic": "Leave Policy", "link": "", "docs": ["leave-v1.pdf", "leave-v2.pdf"]}]
    )

    remove_document_from_topic(str(metadata_file), "Leave Policy", "leave-v1.pdf")

    topics = load_doc_topics(str(metadata_file))
    assert topics[0].docs_filename == ["leave-v2.pdf"]


def test_remove_document_from_topic_is_a_noop_when_not_found(tmp_path):
    metadata_file = tmp_path / "docs-metadata.json"
    _write_metadata(metadata_file, [{"topic": "Leave Policy", "link": "", "docs": ["leave-v1.pdf"]}])

    remove_document_from_topic(str(metadata_file), "Unknown Topic", "leave-v1.pdf")
    remove_document_from_topic(str(metadata_file), "Leave Policy", "nonexistent.pdf")

    topics = load_doc_topics(str(metadata_file))
    assert topics[0].docs_filename == ["leave-v1.pdf"]
