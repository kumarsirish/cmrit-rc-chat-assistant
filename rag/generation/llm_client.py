"""Wrapper around LangChain's ChatOllama.

Keeps all Ollama-specific code (host, model name, request shape) in
one place so the model can be swapped later by editing config.yaml
only, and so the rest of the app (rag_pipeline.py) only ever calls
generate_answer() and never touches ChatOllama directly.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

_ROLE_TO_MESSAGE_CLASS = {"user": HumanMessage, "assistant": AIMessage}


class LlmGenerationError(Exception):
    """Raised when the Ollama server is unreachable, times out, or
    otherwise fails to produce a reply."""


class LlmClient:
    """Sends chat requests to a locally running Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: int,
        num_thread: int,
        num_predict: int = None,
    ):
        self._llm = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_thread=num_thread,
            num_predict=num_predict,
            client_kwargs={"timeout": timeout_seconds},
        )

    def generate_answer(
        self, system_prompt: str, user_prompt: str, conversation_history: list = None
    ) -> str:
        """Sends a system prompt, optional prior turns, and the new user
        prompt, and returns the reply text.

        conversation_history, if given, is a list of
        {"role": "user"/"assistant", "content": str} dicts from earlier
        turns in the same session, placed between the system prompt and
        the new question so the model can follow up on prior context.
        """
        messages = [SystemMessage(content=system_prompt)]
        for turn in conversation_history or []:
            message_class = _ROLE_TO_MESSAGE_CLASS[turn["role"]]
            messages.append(message_class(content=turn["content"]))
        messages.append(HumanMessage(content=user_prompt))

        try:
            response = self._llm.invoke(messages)
        except Exception as exc:
            logger.error("Ollama call failed (model=%s): %s", self._llm.model, exc)
            raise LlmGenerationError(f"Ollama call failed: {exc}") from exc
        return response.content
