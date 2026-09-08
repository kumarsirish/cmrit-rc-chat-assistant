from rag.indexing.docs_metadata import DocTopic, build_filename_topic_map, find_matching_topic


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


def test_find_matching_topic_picks_best_keyword_overlap():
    topics = _sample_topics()
    matched = find_matching_topic("what is the leave application process", topics)
    assert matched.topic == "Leave Information, Application and Approval Process"


def test_find_matching_topic_returns_none_when_no_overlap():
    topics = _sample_topics()
    assert find_matching_topic("what is the weather today", topics) is None


def test_find_matching_topic_matches_link_only_entry():
    topics = _sample_topics()
    matched = find_matching_topic("how do I access the cmrit website", topics)
    assert matched.topic == "CMRIT Website"
    assert matched.docs_filename is None


def test_find_matching_topic_prefers_specific_topic_over_longer_umbrella_topic():
    # Regression test: "leave" is the only shared keyword with both
    # the specific "Leave Information..." topic and a long umbrella
    # "HOD Handbook (... leave information ...)" topic that happens to
    # mention "leave information" as one of a dozen subjects it
    # covers. The umbrella topic's name is much longer, but "leave"
    # covers a far smaller fraction of its own keywords, so it must
    # not win the match.
    topics = [
        DocTopic(
            topic="Leave Information, Application and Approval Process",
            link="https://example.com/leave",
            docs_filename="CMRIT - Leave Information.pdf",
        ),
        DocTopic(
            topic=(
                "HOD Handbook (Initiatives, Teaching & Learning, Mentoring, "
                "Policies, leave information, Faculty evaluation, Department "
                "Evaluation, Exam conduction, Facilities etc.)"
            ),
            link="https://example.com/hod-handbook",
            docs_filename="HoD Handbook - CMRIT.pdf",
        ),
    ]
    matched = find_matching_topic("how to get vacation leave approved", topics)
    assert matched.topic == "Leave Information, Application and Approval Process"


def test_build_filename_topic_map_normalizes_and_skips_link_only():
    topics = _sample_topics()
    filename_map = build_filename_topic_map(topics)
    assert len(filename_map) == 2
    assert filename_map["cmrit - leave information.pdf"].topic == (
        "Leave Information, Application and Approval Process"
    )
    assert "cmrit website" not in filename_map
