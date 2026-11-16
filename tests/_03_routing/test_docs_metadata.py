import json

from rag._03_routing.docs_metadata import DocTopic, build_filename_topic_map, load_doc_topics


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
