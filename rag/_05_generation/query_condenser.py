"""Rewrites a follow-up question into a standalone retrieval query
when it depends on the immediately preceding exchange in the session
— otherwise returns the question unchanged.

This exists because retrieval and generation depend on conversation
history very differently (see rag_pipeline.answer): the answer LLM
gets the full capped history and can resolve "it"/"that leave type"
itself, but dense/BM25 retrieval only ever sees whatever text it's
given directly — a bare "How many days is that?" embeds/searches to
nothing useful on its own. condense() bridges that gap for the
retrieval query only, the same way query_expander.expand_query bridges
the glossary/jargon gap; the original question is untouched.

Deliberately narrow, by design, not oversight:

- Sees only the single most recent exchange (last two entries of
  conversation_history), never the full history window — an unrelated
  topic from several turns back must not leak into today's retrieval
  query just because it's still inside the session's history cap.
- One LLM call both decides whether the new question depends on that
  exchange and produces the rewrite, instead of a separate
  classify-then-rewrite round trip.
- A question that already stands on its own — including one about a
  brand new, unrelated topic — must come back verbatim, unchanged.
- Any failure (LLM error, empty reply) falls back to the original
  question, same as rephrase_suggester does for its LLM call: a
  condensation failure degrades to today's (history-blind) retrieval
  behavior, never blocks the turn.
"""

from rag._05_generation.llm_client import LlmClient, generate_best_effort

_SYSTEM_PROMPT = (
    "You are given the single most recent exchange from an ongoing "
    "conversation, followed by a new question from the same user.\n\n"
    "Decide whether the new question can only be understood by "
    "referring to that prior exchange -- e.g. it uses a pronoun "
    "('it', 'that'), an implicit reference ('the same policy', 'what "
    "about X instead'), or otherwise omits a subject that was only "
    "established in the prior exchange.\n\n"
    "If it depends on the prior exchange: rewrite it into a "
    "standalone question that spells out the specific detail it was "
    "referring to, so it can be understood with no other context. Do "
    "not answer it, and do not add anything beyond what the new "
    "question asked.\n\n"
    "If it does NOT depend on the prior exchange -- it already "
    "stands on its own, even if it happens to be about a related or "
    "a completely unrelated topic -- reply with the new question "
    "copied back exactly, verbatim, unchanged.\n\n"
    "Reply with only the resulting question -- no explanation, no "
    "quotation marks, no preamble."
)


def condense(question: str, conversation_history: list, llm_client: LlmClient) -> str:
    """Returns a standalone retrieval query for question, using at
    most the last exchange in conversation_history. Returns question
    unchanged if there's no history to depend on, the LLM call fails,
    or it returns nothing usable.
    """
    if not conversation_history:
        return question

    last_exchange = conversation_history[-2:]
    exchange_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in last_exchange
    )
    user_prompt = f"PRIOR EXCHANGE:\n{exchange_text}\n\nNEW QUESTION: {question}"

    return generate_best_effort(
        llm_client, _SYSTEM_PROMPT, user_prompt, fallback=question, label="Query condensation"
    )
