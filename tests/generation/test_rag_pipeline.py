"""Integration tests for RagPipeline itself: that it wires retrieval,
context building, the LLM call, and response processing together
correctly. Unit-level behavior of each of those steps has its own test
file (test_context_builder.py, test_response_processor.py).
"""

from rag.generation.rag_pipeline import NO_MATCH_MESSAGE, RagPipeline
from rag.retrieval.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_query = None

    def retrieve(self, question):
        self.last_query = question
        return self._chunks

    def match_topic(self, question):
        return None


class FakeLlmClient:
    def __init__(self, reply):
        self._reply = reply
        self.last_user_prompt = None
        self.last_conversation_history = None

    def generate_answer(self, system_prompt, user_prompt, conversation_history=None):
        self.last_user_prompt = user_prompt
        self.last_conversation_history = conversation_history
        return self._reply


def test_answer_refuses_without_calling_llm_when_no_chunks_found():
    retriever = FakeRetriever(chunks=[])
    llm_client = FakeLlmClient(reply="should not be used")

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
    result = pipeline.answer("What is the capital of France?")

    assert result.answer_text == NO_MATCH_MESSAGE
    assert result.sources == []
    assert result.grounded is False
    assert llm_client.last_user_prompt is None


def test_answer_is_grounded_and_includes_sources_when_chunks_found():
    chunks = [
        RetrievedChunk(
            text="Staff must apply for leave at least 3 days in advance.",
            source_file="leave.pdf",
            page_number=2,
            relevance_score=0.82,
        )
    ]
    retriever = FakeRetriever(chunks=chunks)
    llm_client = FakeLlmClient(reply="Staff must apply for leave 3 days in advance [1].")

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
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
    retriever = FakeRetriever(chunks=chunks)
    llm_client = FakeLlmClient(reply="Yes, the same rule applies.")
    history = [
        {"role": "user", "content": "How far in advance should I apply for leave?"},
        {"role": "assistant", "content": "At least 3 days in advance."},
    ]

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
    pipeline.answer("What about sick leave?", conversation_history=history)

    # The LLM sees the full conversation history ...
    assert llm_client.last_conversation_history == history
    # ... but the retriever only ever sees the current question, never
    # the conversation history — retrieval is deterministic and
    # session-independent.
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
    retriever = FakeRetriever(chunks=chunks)
    llm_client = FakeLlmClient(reply="Staff must apply for leave 3 days in advance.")

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
    pipeline.answer("How far in advance should I apply for leave?")

    assert llm_client.last_conversation_history is None
    assert retriever.last_query == "How far in advance should I apply for leave?"
