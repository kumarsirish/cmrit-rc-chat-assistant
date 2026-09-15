from rag.routing.docs_metadata import DocTopic, build_filename_topic_map


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
