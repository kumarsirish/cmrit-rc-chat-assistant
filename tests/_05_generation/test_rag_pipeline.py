"""Integration tests for RagPipeline itself: that it wires retrieval,
context building, the LLM call, and response processing together
correctly. Unit-level behavior of each of those steps has its own test
file (test_context_builder.py, test_response_processor.py).
"""

from rag._05_generation.rag_pipeline import NO_MATCH_MESSAGE, RagPipeline
from rag._04_retrieval.retrieval_service import RetrievedChunk


class FakeRetrievalService:
    def __init__(self, chunks, matched_topics=None):
        self._chunks = chunks
        self._matched_topics = matched_topics or []
        self.last_query = None
        self.last_trace = {"fake": "trace"}

    def retrieve(self, question, top_k=None, min_reranker_score=None, category_override=None):
        self.last_query = question
        return self._chunks, self._matched_topics


class FakeLlmClient:
    """reply may be a single string, reused for every call (fine when a
    test only cares about the final answer/one call), or a list
    consumed in call order (needed when a test makes more than one
    LLM call, e.g. a rephrase-suggestion call followed by the
    answer-generation call).
    """

    def __init__(self, reply):
        self._replies = reply if isinstance(reply, list) else None
        self._reply = None if self._replies else reply
        self.calls = []
        self.last_user_prompt = None
        self.last_conversation_history = None
        self.call_count = 0

    def generate_answer(self, system_prompt, user_prompt, conversation_history=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "conversation_history": conversation_history,
            }
        )
        self.last_user_prompt = user_prompt
        self.last_conversation_history = conversation_history
        reply = self._replies[self.call_count] if self._replies is not None else self._reply
        self.call_count += 1
        return reply


def test_answer_refuses_without_an_answer_llm_call_when_no_chunks_found():
    retriever = FakeRetrievalService(chunks=[])
    # Used only for the rephrase-suggestion call (see
    # test_answer_offers_a_rephrase_suggestion_when_nothing_is_found),
    # never for generating an answer — there's no context to answer
    # from.
    llm_client = FakeLlmClient(reply="A rephrased question?")

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("What is the capital of France?")

    assert result.answer_text == NO_MATCH_MESSAGE
    assert result.sources == []
    assert result.grounded is False


def test_answer_offers_a_rephrase_suggestion_when_nothing_is_found():
    retriever = FakeRetrievalService(chunks=[])
    llm_client = FakeLlmClient(reply="What is CMRIT's official leave policy?")

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("how many leaves do i get")

    assert result.rephrase_suggestion == "What is CMRIT's official leave policy?"
    assert llm_client.call_count == 1
    assert llm_client.last_user_prompt == "how many leaves do i get"


def test_answer_skips_rephrase_suggestion_when_suggest_rephrase_is_false():
    """A retry of a previous rephrase suggestion (see
    dashboard.render_rephrase_suggestion/render_chat_page) must not
    chain into offering yet another rephrase if it also fails."""
    retriever = FakeRetrievalService(chunks=[])
    llm_client = FakeLlmClient(reply="A rephrased question that also fails?")

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("What is CMRIT's official leave policy?", suggest_rephrase=False)

    assert result.answer_text == NO_MATCH_MESSAGE
    assert result.rephrase_suggestion is None
    assert llm_client.call_count == 0


def test_answer_has_no_rephrase_suggestion_when_topic_matched_without_a_document():
    from dataclasses import dataclass

    @dataclass
    class FakeMatchedTopic:
        topic: str
        link: str
        docs_filename: str = None

    retriever = FakeRetrievalService(chunks=[], matched_topics=[FakeMatchedTopic("Anti-ragging", "https://example.com")])
    llm_client = FakeLlmClient(reply="should not be used")

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("What is the anti-ragging policy?")

    assert result.rephrase_suggestion is None
    assert llm_client.call_count == 0


def test_answer_is_grounded_and_includes_sources_when_chunks_found():
    chunks = [
        RetrievedChunk(
            text="Staff must apply for leave at least 3 days in advance.",
            source_file="leave.pdf",
            page_number=2,
            relevance_score=0.82,
        )
    ]
    retriever = FakeRetrievalService(chunks=chunks)
    llm_client = FakeLlmClient(reply="Staff must apply for leave 3 days in advance [1].")

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("How far in advance should I apply for leave?")

    assert result.grounded is True
    assert len(result.sources) == 1
    assert result.sources[0].source_file == "leave.pdf"
    assert result.sources[0].page_number == 2
    assert result.cited_indices == [1]
    assert "leave.pdf" in llm_client.last_user_prompt


def test_answer_forwards_conversation_history_to_the_llm_but_not_to_retrieval():
    chunks = [
        RetrievedChunk(
            text="Staff must apply for leave at least 3 days in advance.",
            source_file="leave.pdf",
            page_number=2,
            relevance_score=0.82,
        )
    ]
    retriever = FakeRetrievalService(chunks=chunks)
    llm_client = FakeLlmClient(reply="Some answer.")
    history = [
        {"role": "user", "content": "How far in advance should I apply for leave?"},
        {"role": "assistant", "content": "At least 3 days in advance."},
    ]

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    pipeline.answer("What about sick leave?", conversation_history=history)

    # The answer-generation call sees the full conversation history ...
    assert llm_client.last_conversation_history == history
    assert "What about sick leave?" in llm_client.last_user_prompt
    # ... but retrieval only ever sees the question itself, verbatim —
    # there is no follow-up query rewriting.
    assert retriever.last_query == "What about sick leave?"


def test_answer_defaults_to_no_conversation_history():
    chunks = [
        RetrievedChunk(
            text="Staff must apply for leave at least 3 days in advance.",
            source_file="leave.pdf",
            page_number=2,
            relevance_score=0.82,
        )
    ]
    retriever = FakeRetrievalService(chunks=chunks)
    llm_client = FakeLlmClient(reply="Staff must apply for leave 3 days in advance.")

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    pipeline.answer("How far in advance should I apply for leave?")

    assert llm_client.last_conversation_history is None
    assert retriever.last_query == "How far in advance should I apply for leave?"
    assert llm_client.call_count == 1
