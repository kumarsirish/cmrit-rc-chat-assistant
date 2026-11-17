"""Integration tests for RagPipeline itself: that it wires retrieval,
context building, the LLM call, and response processing together
correctly. Unit-level behavior of each of those steps has its own test
file (test_context_builder.py, test_response_processor.py).
"""

from rag._05_generation.rag_pipeline import NO_MATCH_MESSAGE, RagPipeline
from rag._04_retrieval.retrieval_service import RetrievedChunk


class FakeRetrievalService:
    def __init__(self, chunks, matched_topic=None):
        self._chunks = chunks
        self._matched_topic = matched_topic
        self.last_query = None

    def retrieve(self, question, top_k=None, min_reranker_score=None, category_override=None):
        self.last_query = question
        return self._chunks, self._matched_topic


class FakeLlmClient:
    """reply may be a single string, reused for every call (fine when a
    test only cares about the final answer/one call), or a list
    consumed in call order (needed when a test must give a different
    reply to the condensation call vs. the answer-generation call —
    see the query_condenser tests below).
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

    retriever = FakeRetrievalService(chunks=[], matched_topic=FakeMatchedTopic("Anti-ragging", "https://example.com"))
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


def test_answer_forwards_conversation_history_to_the_llm_regardless_of_condensation():
    chunks = [
        RetrievedChunk(
            text="Staff must apply for leave at least 3 days in advance.",
            source_file="leave.pdf",
            page_number=2,
            relevance_score=0.82,
        )
    ]
    retriever = FakeRetrievalService(chunks=chunks)
    llm_client = FakeLlmClient(reply=["How many days of sick leave are allowed?", "Some answer."])
    history = [
        {"role": "user", "content": "How far in advance should I apply for leave?"},
        {"role": "assistant", "content": "At least 3 days in advance."},
    ]

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    pipeline.answer("What about sick leave?", conversation_history=history)

    # The answer-generation call (the last one made) sees the full
    # conversation history — the original question, never a condensed
    # form, is what's sent alongside it (invariant: question is never
    # modified).
    assert llm_client.last_conversation_history == history
    assert "What about sick leave?" in llm_client.last_user_prompt


def test_answer_condenses_retrieval_query_when_question_depends_on_prior_exchange():
    chunks = [
        RetrievedChunk(
            text="Sick leave requires a medical certificate for more than 2 days.",
            source_file="leave.pdf",
            page_number=3,
            relevance_score=0.8,
        )
    ]
    retriever = FakeRetrievalService(chunks=chunks)
    llm_client = FakeLlmClient(
        reply=["How many days of sick leave are employees allowed?", "Some answer [1]."]
    )
    history = [
        {"role": "user", "content": "How many days of casual leave do I get?"},
        {"role": "assistant", "content": "12 days of casual leave per year."},
    ]

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("What about sick leave?", conversation_history=history)

    # Retrieval sees the condensed, standalone form ...
    assert retriever.last_query == "How many days of sick leave are employees allowed?"
    assert result.used_conversation_context is True
    # ... but the condenser call itself only saw the single most recent
    # exchange, not any earlier turns (there aren't any here, but the
    # call is still made with conversation_history=None on the
    # LlmClient side, distinguishing it from the answer-generation call).
    assert llm_client.calls[0]["conversation_history"] is None
    assert llm_client.calls[1]["conversation_history"] == history


def test_answer_leaves_retrieval_query_unchanged_for_a_self_contained_question():
    """A question that doesn't depend on the prior exchange must come
    back from the condenser verbatim — this is the case that protects
    an unrelated follow-up from having a prior, unrelated topic folded
    into its retrieval query (see query_condenser's docstring)."""
    chunks = [
        RetrievedChunk(
            text="The probation period is 1 year from the date of joining.",
            source_file="probation.pdf",
            page_number=1,
            relevance_score=0.85,
        )
    ]
    retriever = FakeRetrievalService(chunks=chunks)
    # The condenser correctly judges the new question self-contained
    # and returns it unchanged; only the second (answer) call gets a
    # distinct reply.
    llm_client = FakeLlmClient(
        reply=["What is the probation period completion process?", "Some answer [1]."]
    )
    history = [
        {"role": "user", "content": "How many days of casual leave do I get?"},
        {"role": "assistant", "content": "12 days of casual leave per year."},
    ]

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("What is the probation period completion process?", conversation_history=history)

    assert retriever.last_query == "What is the probation period completion process?"
    assert result.used_conversation_context is False


def test_answer_falls_back_to_original_question_when_condensation_fails():
    from rag._05_generation.llm_client import LlmGenerationError

    class FailingCondenseThenAnswerLlmClient(FakeLlmClient):
        def generate_answer(self, system_prompt, user_prompt, conversation_history=None):
            if conversation_history is None:
                raise LlmGenerationError("condensation backend down")
            return super().generate_answer(system_prompt, user_prompt, conversation_history)

    chunks = [
        RetrievedChunk(
            text="Sick leave requires a medical certificate for more than 2 days.",
            source_file="leave.pdf",
            page_number=3,
            relevance_score=0.8,
        )
    ]
    retriever = FakeRetrievalService(chunks=chunks)
    llm_client = FailingCondenseThenAnswerLlmClient(reply="Some answer [1].")
    history = [
        {"role": "user", "content": "How many days of casual leave do I get?"},
        {"role": "assistant", "content": "12 days of casual leave per year."},
    ]

    pipeline = RagPipeline(retrieval_service=retriever, llm_client=llm_client)
    result = pipeline.answer("What about sick leave?", conversation_history=history)

    assert retriever.last_query == "What about sick leave?"
    assert result.grounded is True


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
    # No conversation history at all means query_condenser is never
    # called — only the one answer-generation call happens.
    assert llm_client.call_count == 1
