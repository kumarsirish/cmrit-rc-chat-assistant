from rag._05_generation.llm_client import LlmGenerationError
from rag._05_generation.rephrase_suggester import suggest_rephrase


class FakeLlmClient:
    def __init__(self, reply):
        self._reply = reply
        self.last_user_prompt = None

    def generate_answer(self, system_prompt, user_prompt, conversation_history=None):
        self.last_user_prompt = user_prompt
        if self._reply is None:
            raise LlmGenerationError("boom")
        return self._reply


def test_suggest_rephrase_passes_through_known_glossary_acronyms():
    glossary = {"RCPIE": "Research, Consultancy, Projects, Innovation and Entrepreneurship"}
    llm_client = FakeLlmClient(reply="What is the RCPIE load distribution per department?")

    suggest_rephrase("What is the load distribution of RCPIE for a department", llm_client, glossary)

    # The LLM must be given the real expansion, not left to guess one —
    # see the module docstring for the hallucination this prevents.
    assert "RCPIE = Research, Consultancy, Projects, Innovation and Entrepreneurship" in llm_client.last_user_prompt


def test_suggest_rephrase_omits_known_acronyms_block_when_none_match():
    glossary = {"RCPIE": "Research, Consultancy, Projects, Innovation and Entrepreneurship"}
    llm_client = FakeLlmClient(reply="A rephrased question?")

    suggest_rephrase("How many days of casual leave do I get?", llm_client, glossary)

    assert "KNOWN ACRONYMS" not in llm_client.last_user_prompt


def test_suggest_rephrase_works_without_a_glossary():
    llm_client = FakeLlmClient(reply="A rephrased question?")

    result = suggest_rephrase("What is the RCPIE policy?", llm_client, glossary=None)

    assert result == "A rephrased question?"
    assert "KNOWN ACRONYMS" not in llm_client.last_user_prompt


def test_suggest_rephrase_matches_acronyms_case_insensitively():
    glossary = {"RCPIE": "Research, Consultancy, Projects, Innovation and Entrepreneurship"}
    llm_client = FakeLlmClient(reply="A rephrased question?")

    suggest_rephrase("what is the rcpie policy", llm_client, glossary)

    assert "RCPIE = Research, Consultancy, Projects, Innovation and Entrepreneurship" in llm_client.last_user_prompt


def test_suggest_rephrase_returns_none_when_llm_fails():
    llm_client = FakeLlmClient(reply=None)
    assert suggest_rephrase("What about sick leave?", llm_client) is None


def test_suggest_rephrase_returns_none_when_unchanged():
    llm_client = FakeLlmClient(reply="What about sick leave?")
    assert suggest_rephrase("What about sick leave?", llm_client) is None
