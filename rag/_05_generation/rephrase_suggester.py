"""Suggests a better-phrased question when the pipeline found nothing.

When retrieval or generation comes up empty (see response_processor.
build_no_result_response / RagAnswer.grounded), the user's own wording
is often the reason nothing matched — policy documents use their own
formal vocabulary that a casually-phrased question won't share. Rather
than leave the user to guess a better phrasing themselves, this asks
the LLM to write a HyDE-style rephrasing: not a paraphrase of the
question, but a guess at how the *answer* would actually be worded in
a policy document, turned back into a question that uses that same
vocabulary. That tends to share more words/phrasing with the indexed
text than the original question did, giving the user a concrete
next thing to try instead of a bare "no results" dead end.

This only ever produces a suggestion shown in the UI (see
dashboard.render_rephrase_suggestion) — it is never fed back into
retrieval automatically, so one no-result answer can't spiral into a
chain of silent, unseen retries.
"""

import logging

from rag._05_generation.llm_client import LlmClient, LlmGenerationError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "A user asked a question about CMRIT college policies, and no "
    "matching information was found in the indexed policy documents. "
    "Write ONE alternative phrasing of the same question that is more "
    "likely to match how a formal college policy document would "
    "phrase the topic — e.g. using the likely official policy/section "
    "name, more formal or specific terminology, or spelling out an "
    "acronym the user may have abbreviated or vice versa. "
    "Keep it a single question asking about the same underlying topic "
    "— do not change what is being asked about, do not answer it, and "
    "do not add any preamble, quotation marks, or explanation. Reply "
    "with only the rephrased question itself."
)


def suggest_rephrase(question: str, llm_client: LlmClient) -> str:
    """Returns one alternative phrasing of question, or None if the LLM
    call fails — a missing suggestion is never treated as fatal, since
    the original no-result answer has already been produced either
    way (see rag_pipeline.answer)."""
    try:
        rephrased = llm_client.generate_answer(_SYSTEM_PROMPT, question)
    except LlmGenerationError:
        logger.warning("Rephrase suggestion failed for question: %r", question, exc_info=True)
        return None
    rephrased = rephrased.strip().strip('"')
    if not rephrased or rephrased.lower() == question.strip().lower():
        return None
    return rephrased
