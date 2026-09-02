"""Wrapper around the Ollama chat API.

Keeps all Ollama-specific code (host, model name, request shape) in
one place so the model can be swapped later by editing config.yaml
only.
"""

import ollama


class LlmClient:
    """Sends chat requests to a locally running Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: int,
        num_thread: int,
    ):
        self._model = model
        self._temperature = temperature
        self._num_thread = num_thread
        self._client = ollama.Client(host=base_url, timeout=timeout_seconds)

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
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat(
            model=self._model,
            messages=messages,
            options={
                "temperature": self._temperature,
                "num_thread": self._num_thread,
            },
        )
        return response["message"]["content"]
