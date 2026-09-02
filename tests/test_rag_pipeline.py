from rag.rag_pipeline import NO_MATCH_MESSAGE, RagPipeline
from rag.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_query = None

    def retrieve(self, question):
        self.last_query = question
        return self._chunks


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
    llm_client = FakeLlmClient(reply="Staff must apply for leave 3 days in advance.")

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
    result = pipeline.answer("How far in advance should I apply for leave?")

    assert result.grounded is True
    assert len(result.sources) == 1
    assert result.sources[0].source_file == "leave.pdf"
    assert result.sources[0].page_number == 2
    assert "leave.pdf" in llm_client.last_user_prompt


def test_answer_forwards_conversation_history_to_the_llm():
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

    assert llm_client.last_conversation_history == history


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


def test_answer_strips_inline_source_citations_from_the_llm_reply():
    chunks = [
        RetrievedChunk(
            text="Staff must apply for leave at least 3 days in advance.",
            source_file="leave.pdf",
            page_number=2,
            relevance_score=0.82,
        )
    ]
    retriever = FakeRetriever(chunks=chunks)
    llm_client = FakeLlmClient(
        reply=(
            "Staff must apply at least 3 days in advance.\n\n"
            "[Source 1: leave.pdf, page 2]"
        )
    )

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
    result = pipeline.answer("How far in advance should I apply for leave?")

    assert result.answer_text == "Staff must apply at least 3 days in advance."
    assert "[Source" not in result.answer_text
    # The sources list (used to render links below the answer) is untouched.
    assert result.sources[0].source_file == "leave.pdf"


def test_answer_strips_unbracketed_prose_citations_from_the_llm_reply():
    # Reproduces the reported case: the model cited the source in plain
    # prose, with a "CONTEXT" lead-in, and no brackets at all.
    chunks = [
        RetrievedChunk(
            text="TYL stands for Tie Your LACES.",
            source_file="HoD Handbook - CMRIT.pdf",
            page_number=20,
            relevance_score=0.7,
        )
    ]
    retriever = FakeRetriever(chunks=chunks)
    llm_client = FakeLlmClient(
        reply=(
            "According to the provided CONTEXT, specifically from Source 1: "
            'HoD Handbook - CMRIT.pdf, page 20, TYL stands for "Tie Your LACES".'
        )
    )

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
    result = pipeline.answer("what is TYL")

    assert result.answer_text == 'TYL stands for "Tie Your LACES".'
    assert "Source" not in result.answer_text
    assert "CONTEXT" not in result.answer_text


def test_retrieval_query_is_broadened_with_the_previous_user_turn():
    chunks = [
        RetrievedChunk(
            text="TYL stands for Tie Your LACES.",
            source_file="HoD Handbook - CMRIT.pdf",
            page_number=20,
            relevance_score=0.7,
        )
    ]
    retriever = FakeRetriever(chunks=chunks)
    llm_client = FakeLlmClient(reply="Some more detail about TYL.")
    history = [
        {"role": "user", "content": "what is TYL"},
        {"role": "assistant", "content": 'TYL stands for "Tie Your LACES".'},
    ]

    pipeline = RagPipeline(retriever=retriever, llm_client=llm_client)
    pipeline.answer("give details", conversation_history=history)

    assert retriever.last_query == "what is TYL give details"


def test_retrieval_query_is_unchanged_without_conversation_history():
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

    assert retriever.last_query == "How far in advance should I apply for leave?"
