"""Model-agnostic wrapper around LangChain chat models.

Which provider answers real user questions is selected by config.yaml's
llm.provider ("ollama" or "groq"); LlmClient builds the right LangChain
chat model via init_chat_model() and hides the provider-specific setup
(base URL vs. API key, num_predict vs. max_tokens, ...) behind it. The
rest of the app (rag_pipeline.py) only ever calls generate_answer() and
never touches the underlying chat model or provider directly, so adding
another LangChain-supported provider later means adding one branch to
_build_provider_kwargs() here — nothing else changes.
"""

import logging
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_ROLE_TO_MESSAGE_CLASS = {"user": HumanMessage, "assistant": AIMessage}


class LlmGenerationError(Exception):
    """Raised when the configured LLM provider is unreachable, times
    out, misconfigured (e.g. missing API key), or otherwise fails to
    produce a reply."""


def _build_provider_kwargs(
    provider: str, temperature: float, timeout_seconds: int, provider_settings: dict
) -> dict:
    """Translates the provider-neutral args plus config.yaml's
    llm.<provider> block into the keyword arguments init_chat_model()
    should pass through to that provider's LangChain chat model."""
    if provider == "ollama":
        return {
            "base_url": provider_settings["base_url"],
            "temperature": temperature,
            "num_thread": provider_settings.get("num_thread"),
            "num_predict": provider_settings.get("num_predict"),
            "client_kwargs": {"timeout": timeout_seconds},
        }

    if provider == "groq":
        api_key_env = provider_settings.get("api_key_env", "GROQ_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LlmGenerationError(
                f"Groq API key not found: environment variable {api_key_env!r} "
                "is unset or empty."
            )
        return {
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": provider_settings.get("max_tokens"),
            "timeout": timeout_seconds,
        }

    raise LlmGenerationError(
        f"Unsupported LLM provider {provider!r}. Supported providers: ollama, groq."
    )


class LlmClient:
    """Sends chat requests to whichever LLM provider config.yaml's
    llm.provider selects."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        timeout_seconds: int,
        **provider_settings,
    ):
        self._provider = provider
        self._model = model
        kwargs = _build_provider_kwargs(provider, temperature, timeout_seconds, provider_settings)
        self._llm = init_chat_model(model=model, model_provider=provider, **kwargs)

    @classmethod
    def from_config(cls, llm_config, temperature: float = None) -> "LlmClient":
        """Builds the client for whichever provider llm_config.provider
        selects, reading that provider's settings block (llm_config.ollama
        / llm_config.groq / ...) off the same AppConfig.llm object.

        temperature, if given, overrides that provider's configured
        temperature (used by the UI's temperature slider).

        This is the one place that needs to know the provider-config
        blocks share a (model, temperature, request_timeout_seconds, ...)
        shape — callers just pass config.llm.
        """
        provider = llm_config.provider
        provider_cfg = getattr(llm_config, provider)
        return cls(
            provider=provider,
            model=provider_cfg.model,
            temperature=provider_cfg.temperature if temperature is None else temperature,
            timeout_seconds=provider_cfg.request_timeout_seconds,
            **{
                field: value
                for field, value in vars(provider_cfg).items()
                if field not in ("model", "temperature", "request_timeout_seconds")
            },
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
            logger.error(
                "LLM call failed (provider=%s, model=%s): %s", self._provider, self._model, exc
            )
            raise LlmGenerationError(f"{self._provider} call failed: {exc}") from exc
        return response.content
