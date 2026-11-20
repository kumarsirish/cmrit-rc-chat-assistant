"""Suggests a better-phrased question when the pipeline found nothing.

When retrieval or generation comes up empty (see response_processor.
build_no_result_response / RagAnswer.grounded), the user's own wording
is often the reason nothing matched. Rather than leave the user to
guess a better phrasing themselves, this asks the LLM for ONE
alternative phrasing — kept close to how a person would naturally ask
it, same core terms as the original, not formal document/policy-manual
language.

This deliberately replaced an earlier HyDE-style design that asked the
LLM to guess how the *answer* would be worded in a policy document
(formal section-name vocabulary, "designated"/"official"-type
language). That backfired in practice: a real case where the original
question failed, the HyDE-style rephrasing actually retrieved good,
relevant chunks, and STILL got refused — the rephrasing's formal
wording ("designated percentage... allocated to research activities")
implied the document states one specific, explicitly-labeled figure,
which the retrieved text (general, proportional-to-contribution
language) didn't literally satisfy, so the strict grounding check
correctly refused that overclaimed framing even with the right content
in hand. A plainer, looser rephrasing the user tried by hand
("what percentage... is for research") worked. The lesson: a
rephrasing should never be MORE specific or definite than the
original question — only fix the likely actual blocker (ambiguous
wording, an abbreviated acronym), not add implied precision the
original never asked for.

This only ever produces a suggestion shown in the UI (see
dashboard.render_rephrase_suggestion) — it is never fed back into
retrieval automatically, so one no-result answer can't spiral into a
chain of silent, unseen retries.

glossary (see rag._01_ingestion.glossary_extractor), if given, is
checked for any acronym literally present in the question and its real
expansion is handed to the LLM as a KNOWN ACRONYMS block — without
this, asked to spell out an acronym, the model has nothing to ground
that in but its own guess, and will confidently invent a wrong
expansion for an institution-specific acronym it's never seen (e.g.
"RCPIE" guessed as "Regional Civil Police Inspectorate" instead of the
glossary's real "Research, Consultancy, Projects, Innovation and
Entrepreneurship").
"""

import re

from rag._05_generation.llm_client import LlmClient, generate_best_effort

_SYSTEM_PROMPT = (
    "A user asked a question and no matching information was found in "
    "the indexed documents. "
    "Write ONE alternative phrasing of the same question, the way a "
    "person would naturally ask it out loud — not formal document or "
    "policy-manual language. Keep the same core terms the original "
    "question used; only change wording that's likely the actual "
    "blocker, e.g. spelling out an acronym the user may have "
    "abbreviated (or vice versa), or swapping one ambiguous word for a "
    "clearer one. "
    "Do not make the question more specific or definite than the "
    "original. Do not add words implying the document states one "
    "exact, named, or officially-designated figure or policy (e.g. "
    "'designated', 'official', 'exact') unless the original question "
    "already implied that — an overly specific rephrasing can get "
    "refused even when the underlying content would have answered the "
    "original, looser question. "
    "If a KNOWN ACRONYMS block is given below the question, those are "
    "this organization's own verified acronym meanings — use that "
    "exact expansion for any acronym listed there; never guess a "
    "different one for it. "
    "Keep it a single question asking about the same underlying topic "
    "— do not change what is being asked about, do not answer it, and "
    "do not add any preamble, quotation marks, or explanation. Reply "
    "with only the rephrased question itself."
)

_WORD_PATTERN = re.compile(r"[A-Za-z]+")


def _known_acronyms_in(question: str, glossary: dict) -> dict:
    """Returns {acronym: phrase} for every glossary acronym literally
    present in question as a whole word (case-insensitive)."""
    if not glossary:
        return {}
    words = {word.upper() for word in _WORD_PATTERN.findall(question)}
    return {acronym: phrase for acronym, phrase in glossary.items() if acronym.upper() in words}


def suggest_rephrase(question: str, llm_client: LlmClient, glossary: dict = None) -> str:
    """Returns one alternative phrasing of question, or None if the LLM
    call fails or comes back unchanged — a missing suggestion is never
    treated as fatal, since the original no-result answer has already
    been produced either way (see rag_pipeline.answer)."""
    known_acronyms = _known_acronyms_in(question, glossary)
    user_prompt = question
    if known_acronyms:
        definitions = "\n".join(f"{acronym} = {phrase}" for acronym, phrase in known_acronyms.items())
        user_prompt = f"{question}\n\nKNOWN ACRONYMS:\n{definitions}"

    rephrased = generate_best_effort(
        llm_client, _SYSTEM_PROMPT, user_prompt, fallback=None, label="Rephrase suggestion"
    )
    if rephrased and rephrased.lower() == question.strip().lower():
        return None
    return rephrased
