from dataclasses import dataclass

from rag._06_response.response_processor import NO_MATCH_MESSAGE, build_no_result_response, process


@dataclass
class FakeMatchedTopic:
    topic: str
    link: str
    docs_filename: str = None


@dataclass
class FakeChunk:
    text: str
    source_file: str
    page_number: int
    relevance_score: float
    match_type: str = "dense"
    source_link: str = ""
    page_end: int = None
    section_title: str = ""
    low_confidence: bool = False

    def __post_init__(self):
        if self.page_end is None:
            self.page_end = self.page_number


def _chunk(**overrides):
    defaults = dict(
        text="Casual leave is 12 days a year.",
        source_file="leave.pdf",
        page_number=4,
        relevance_score=0.9,
    )
    defaults.update(overrides)
    return FakeChunk(**defaults)


def test_build_no_result_response_without_matched_topics():
    result = build_no_result_response(matched_topics=[])
    assert result.answer_text == NO_MATCH_MESSAGE
    assert result.sources == []
    assert result.grounded is False
    assert result.topic_name is None


def test_build_no_result_response_with_a_matched_topic_points_at_its_link():
    topic = FakeMatchedTopic(topic="VTU Evaluation Guidelines", link="https://example.com/vtu")
    result = build_no_result_response(matched_topics=[topic])
    assert "VTU Evaluation Guidelines" in result.answer_text
    assert result.topic_name == "VTU Evaluation Guidelines"
    assert result.topic_link == "https://example.com/vtu"
    assert result.grounded is False


def test_build_no_result_response_with_several_matched_topics_uses_the_first():
    # The best-scoring match (matched_topics[0], already sorted by the
    # matcher) is the one surfaced — several links would read as the
    # system being unsure rather than giving one confident next step.
    best = FakeMatchedTopic(topic="Best Match", link="https://example.com/best")
    other = FakeMatchedTopic(topic="Other Match", link="https://example.com/other")
    result = build_no_result_response(matched_topics=[best, other])
    assert result.topic_name == "Best Match"
    assert result.topic_link == "https://example.com/best"


def test_process_normalizes_verbose_citation_into_short_bracket_form():
    chunks = [_chunk()]
    raw_answer = "Staff must apply at least 3 days in advance.\n\n[Source 1: leave.pdf, page 2]"

    result = process(raw_answer, chunks, matched_topics=[])

    assert result.answer_text == "Staff must apply at least 3 days in advance.[1]"
    assert result.cited_indices == [1]
    assert result.sources[0].source_file == "leave.pdf"


def test_process_normalizes_unbracketed_prose_citation():
    chunks = [_chunk(source_file="HoD Handbook - CMRIT.pdf", page_number=20)]
    raw_answer = (
        "According to the provided CONTEXT, specifically from Source 1: "
        'HoD Handbook - CMRIT.pdf, page 20, TYL stands for "Tie Your LACES".'
    )

    result = process(raw_answer, chunks, matched_topics=[])

    assert result.answer_text == '[1]TYL stands for "Tie Your LACES".'
    assert "Source" not in result.answer_text
    assert "CONTEXT" not in result.answer_text
    assert result.cited_indices == [1]


def test_process_keeps_valid_inline_citation_markers():
    chunks = [_chunk()]
    result = process("Employees are eligible for 12 days of leave [1].", chunks, matched_topics=[])

    assert result.answer_text == "Employees are eligible for 12 days of leave [1]."
    assert result.cited_indices == [1]


def test_process_strips_fabricated_out_of_range_citation():
    # Only one chunk (Source 1) was retrieved; a "[2]" citation has
    # nothing real behind it and must not be presented as one.
    chunks = [_chunk()]
    result = process("Employees are eligible for 12 days of leave [2].", chunks, matched_topics=[])

    assert "[2]" not in result.answer_text
    assert result.cited_indices == []


def test_process_with_no_citation_at_all_still_lists_all_retrieved_sources():
    # A model can answer without citing (e.g. a short non-committal
    # reply); the sources list still reflects everything retrieved so
    # the UI has something to fall back to.
    chunks = [_chunk(), _chunk(source_file="probation.pdf", page_number=1)]
    result = process("General information is available in the handbook.", chunks, matched_topics=[])

    assert result.cited_indices == []
    assert len(result.sources) == 2


def test_process_handles_citations_to_multiple_distinct_sources():
    chunks = [
        _chunk(source_file="leave.pdf", page_number=4),
        _chunk(source_file="probation.pdf", page_number=1),
    ]
    result = process("Leave is 12 days [1]. Probation is 6 months [2].", chunks, matched_topics=[])

    assert result.cited_indices == [1, 2]
    assert result.sources[0].source_file == "leave.pdf"
    assert result.sources[1].source_file == "probation.pdf"


def test_process_deduplicates_repeated_citation_of_the_same_source():
    chunks = [_chunk()]
    result = process("Leave is 12 days [1]. This is confirmed [1].", chunks, matched_topics=[])

    assert result.cited_indices == [1]


def test_process_marks_grounded_false_when_answer_is_the_no_match_message():
    chunks = [_chunk()]
    result = process(NO_MATCH_MESSAGE, chunks, matched_topics=[])

    assert result.grounded is False


def test_process_propagates_low_confidence_from_chunk_to_answer_and_source():
    chunks = [_chunk(low_confidence=True)]
    result = process("Casual leave is 12 days [1].", chunks, matched_topics=[])

    assert result.low_confidence is True
    assert result.sources[0].low_confidence is True


def test_process_low_confidence_is_false_when_no_chunk_is_low_confidence():
    chunks = [_chunk(low_confidence=False)]
    result = process("Casual leave is 12 days [1].", chunks, matched_topics=[])

    assert result.low_confidence is False


def test_process_low_confidence_answer_that_refuses_is_not_flagged():
    # A low-confidence chunk that the LLM decided didn't actually
    # answer the question refuses normally (grounded=False, no
    # sources shown) — low_confidence shouldn't be True for a refusal
    # with nothing behind it to verify.
    chunks = [_chunk(low_confidence=True)]
    result = process(NO_MATCH_MESSAGE, chunks, matched_topics=[])

    assert result.grounded is False
    assert result.low_confidence is False
