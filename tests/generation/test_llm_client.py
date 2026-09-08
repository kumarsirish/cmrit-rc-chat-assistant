import pytest

from rag.generation.llm_client import LlmClient, LlmGenerationError


def _make_client():
    # Constructing ChatOllama doesn't itself connect to a server, so
    # this is safe to build without a running Ollama instance; only
    # .invoke() would actually make a network call.
    return LlmClient(
        provider="ollama",
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0,
        timeout_seconds=5,
        num_thread=1,
    )


def test_generate_answer_wraps_a_failed_ollama_call(monkeypatch):
    client = _make_client()

    def _raise(self, *args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(type(client._llm), "invoke", _raise)

    with pytest.raises(LlmGenerationError, match="ollama call failed"):
        client.generate_answer("system prompt", "user prompt")


def test_generate_answer_returns_reply_content_on_success(monkeypatch):
    client = _make_client()

    class FakeResponse:
        content = "The answer."

    monkeypatch.setattr(type(client._llm), "invoke", lambda self, messages: FakeResponse())

    assert client.generate_answer("system prompt", "user prompt") == "The answer."
