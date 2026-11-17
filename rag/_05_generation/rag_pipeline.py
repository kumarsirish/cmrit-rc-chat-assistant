"""Step 6: Ties retrieval and generation together.

Produces answers that are grounded in the retrieved policy text only.
Two layers keep the assistant from answering with outside knowledge:

1. If retrieval finds no chunk above the relevance threshold, the
   pipeline refuses immediately without calling the LLM for an answer.
2. If chunks are found, the LLM is given a strict system prompt that
   restricts it to the supplied context and told to say so when the
   context does not contain the answer.

Either way, a plain "nothing found" refusal (not the topic-matched
branch, which already gives an official link instead) makes one more
LLM call to suggest a better-phrased question — see
rephrase_suggester and RagAnswer.rephrase_suggestion.

RagPipeline itself only orchestrates: it delegates context formatting
to context_builder and citation/response assembly to
response_processor, so each of those concerns can be read, tested, and
changed independently of the others. Retrieval runs on the current
question alone by default — conversation history is passed to the LLM
(see answer() below) so it can resolve references like "it" in its
reply, but never touches the retrieval query itself, UNLESS
query_condenser decides this question depends on the immediately
preceding exchange, in which case only that one exchange (never the
rest of the session's history) can rewrite the retrieval query — the
question sent to the LLM/UI/citations is untouched either way. See
query_condenser's docstring for the invariants this enforces. This
keeps retrieval deterministic and easy to evaluate/debug independent
of session state, while still letting a genuine follow-up ("how many
days is that?") retrieve something.
"""

import dataclasses
import logging
import time

from rag._05_generation import context_builder
from rag._05_generation import query_condenser
from rag._05_generation import rephrase_suggester
from rag._05_generation.llm_client import LlmClient
from rag._06_response import response_processor
from rag._06_response.response_processor import NO_MATCH_MESSAGE, RagAnswer, SourceReference  # re-exported
from rag._04_retrieval.retrieval_service import RetrievalService
from rag._03_routing import query_expander


def _build_system_prompt(no_match_message: str) -> str:
    """Builds the answer LLM's system prompt, substituting in the
    deployment's configured no-match refusal wording (see
    RagPipeline.__init__/config.app.no_match_message) so the exact
    sentence the LLM is told to reply with verbatim matches whatever
    response_processor.process() checks for (see its no_match_message
    docstring)."""
    return (
        "You are an assistant that answers questions using ONLY the CONTEXT "
        "provided below, which is extracted from the organization's indexed "
        "documents. "
        "Do not use any outside knowledge, even if you believe it to be true. "
        "The CONTEXT sources vary in quality — some are extracted from "
        "slide decks or other messy layouts and may contain stray "
        "fragments, page numbers, or irrelevant text alongside the useful "
        "part. Judge each source independently: if even one source clearly "
        "supports an answer, use it and cite it, even if other sources in "
        "the CONTEXT are noisy, poorly formatted, or unrelated to the "
        "question. Never let noise or irrelevance in one source stop you "
        "from using a clear answer found in another. In particular, some "
        "sources are just a navigation link, table-of-contents row, or "
        "bare mention of a document's name (e.g. a bullet reading only "
        "'Staff Leave Policy' with no other detail) rather than that "
        "document's actual content — this is still noise, not a competing "
        "or more authoritative source. If another source in the CONTEXT "
        "contains that same document's substantive text (the actual rules, "
        "numbers, or definition), answer from that one; do not let a bare "
        "name-only mention make you doubt, hedge on, or refuse to use a "
        "source that genuinely answers the question. "
        "If the CONTEXT does not contain the answer, reply exactly: "
        f'"{no_match_message}" '
        "Answer ONLY the specific question asked — nothing more. Do not "
        "add related rules, exceptions, background, or nearby topics "
        "from the CONTEXT that the question did not ask about, even if "
        "they cover the same document or topic. For example, if asked how "
        "many casual leaves are allowed, state only that number — do not "
        "also describe vacation leave, compensatory-off, or approval "
        "steps unless the question asked about those too. When a detail "
        "in the CONTEXT is a direct precondition for the answer (e.g. an "
        "eligibility requirement the number depends on), include it; "
        "otherwise leave it out. "
        "Keep URLs from the context exactly as written and include them if they are relevant. "
        "Cite your source for every factual claim: immediately after the "
        "sentence, add the bracketed number of the CONTEXT source it came "
        "from, e.g. 'Employees are eligible for 12 days of casual leave "
        "[1].' Use ONLY the numbers given in the CONTEXT's 'Source N' "
        "labels — never invent a number, never cite a source that doesn't "
        "support that specific sentence, and never write the source's "
        "name, page number, or the word 'Source' out in the sentence "
        "itself; the bracketed number is the only citation form. If a "
        "sentence draws on more than one source, cite all of them, e.g. "
        "'[1][2]'. "
        "Earlier turns from this conversation may be included before the "
        "current question purely to resolve references like 'it' or 'that "
        "leave type' — they are not an additional source of facts. Every "
        "factual claim must still come from the CONTEXT given with the "
        "current question. "
        "Format the answer for readability: never write it as one long "
        "paragraph. Break it into short paragraphs of no more than 10 "
        "sentences each, and when the answer describes a sequence of "
        "steps, a process, or a list of items, use numbered steps or "
        "bullet points instead of prose. "
        "A source tagged '— LOW-CONFIDENCE MATCH' scored below the "
        "confident-match threshold during retrieval, but is still the "
        "single best-matching passage found across the entire document "
        "collection — every other candidate scored even lower. Read it "
        "carefully before refusing: source text is often terse or uses "
        "different wording than the question (e.g. a duty listed as a "
        "recurring task, like a weekly requirement, can be the answer to "
        "a 'how often' question even without using the word 'often' or "
        "'frequency'). If it genuinely answers the question, use it and "
        "cite it as normal — the UI already warns the user this match is "
        "low-confidence, so do not add your own caveat sentence about it, "
        "and do not refuse merely because it is tagged low-confidence. "
        "Only refuse if, after that careful reading, the source's actual "
        "subject matter has nothing to do with what was asked. If it does "
        "not answer the question, refuse normally with the "
        "exact refusal sentence above; do not stretch a low-confidence "
        "source to seem more relevant than it is."
    )


logger = logging.getLogger(__name__)


class RagPipeline:
    """Answers a user question grounded in the retrieved policy chunks."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_client: LlmClient,
        glossary: dict = None,
        no_match_message: str = NO_MATCH_MESSAGE,
    ):
        """glossary, if given (see rag._01_ingestion.glossary_extractor),
        expands the retrieval query with matched institutional
        acronyms before topic-matching/dense retrieval — see
        rag._03_routing.query_expander. The index is built once here, not
        per-question, since it never changes between calls.

        no_match_message (normally config.app.no_match_message) is the
        exact refusal sentence the answer LLM is told to reply with
        verbatim when the CONTEXT doesn't answer the question — baked
        into the system prompt once here, rather than per-call, since it
        never changes between calls either."""
        self._retrieval_service = retrieval_service
        self._llm_client = llm_client
        self._glossary = glossary or {}
        self._glossary_index = query_expander.build_glossary_index(self._glossary)
        self._no_match_message = no_match_message
        self._system_prompt = _build_system_prompt(no_match_message)

    def _finalize(
        self,
        rag_answer: RagAnswer,
        question: str,
        suggest_rephrase: bool,
        used_conversation_context: bool,
    ) -> RagAnswer:
        """Attaches an alternative phrasing suggestion (see
        rephrase_suggester) and used_conversation_context (see
        RagAnswer's docstring) — the two bits every answer() return
        point needs stamped onto its result, done here once instead of
        at each of the two return points separately.

        The rephrase suggestion only applies to the plain "nothing
        found" refusal (self._no_match_message) — the topic-matched
        branch of build_no_result_response is also ungrounded but
        already gives the user a concrete official link instead, so a
        rephrasing suggestion there would just be clutter, not a next
        step.

        suggest_rephrase=False (passed by callers retrying a question
        that IS ITSELF a previous rephrase suggestion — see
        rag_pipeline.answer) skips the suggestion entirely, so a
        failed rephrase attempt doesn't chain into offering yet
        another one."""
        if suggest_rephrase and rag_answer.answer_text == self._no_match_message:
            suggestion = rephrase_suggester.suggest_rephrase(question, self._llm_client)
            rag_answer = dataclasses.replace(rag_answer, rephrase_suggestion=suggestion)
        return dataclasses.replace(rag_answer, used_conversation_context=used_conversation_context)

    def answer(
        self,
        question: str,
        conversation_history: list = None,
        top_k: int = None,
        min_reranker_score: float = None,
        suggest_rephrase: bool = True,
        category_override: str = None,
    ) -> RagAnswer:
        """Answers question, grounded in freshly retrieved chunks.

        suggest_rephrase=False skips attaching a rephrase suggestion to
        a "nothing found" refusal — set this by the caller when question
        is itself a retry of a previous rephrase suggestion, so a
        failed retry doesn't chain into offering yet another rephrase
        (see dashboard.render_rephrase_suggestion/render_chat_page).

        category_override, if given (see dashboard.py's Chat page
        category picker), is passed straight through to
        retrieval_service.retrieve() — see its docstring for exactly
        what a concrete category name vs.
        retrieval_service.ALL_CATEGORIES vs. None each do.

        conversation_history (prior {"role", "content"} turns from the
        same session, if any) is passed to the LLM below so it can
        resolve references like "it"/"that" in its reply. It's also
        offered to query_condenser, which rewrites the retrieval query
        (not question itself) IF this question depends on the single
        most recent exchange — see query_condenser's docstring for
        exactly how narrow that is. question (uncondensed, unexpanded)
        is always what's sent to the LLM and everything user-facing,
        so answer phrasing and citations never depend on either step
        below.

        top_k / min_reranker_score, if given, override the retrieval
        service's configured defaults for this call only (used by
        evaluation scripts to try settings without editing config.yaml).

        retrieval_service.retrieve() returns the matched topic along
        with the chunks, so it only has to detect/embed the topic
        match once per question instead of a second time here.

        The retrieval query itself may differ from question — see
        query_condenser.condense (follow-up rewriting) and
        query_expander.expand_query (glossary expansion), applied in
        that order — but only for topic-matching/embedding.
        """
        overall_start = time.perf_counter()
        condensed_question = query_condenser.condense(question, conversation_history, self._llm_client)
        # True iff condense() actually rewrote the query, i.e. this
        # question depended on the immediately preceding exchange —
        # see RagAnswer.used_conversation_context.
        used_conversation_context = condensed_question != question
        retrieval_query = query_expander.expand_query(
            condensed_question, self._glossary, self._glossary_index
        )
        chunks, matched_topic = self._retrieval_service.retrieve(
            retrieval_query, top_k=top_k, min_reranker_score=min_reranker_score,
            category_override=category_override,
        )
        retrieval_seconds = time.perf_counter() - overall_start

        if not chunks:
            total_seconds = time.perf_counter() - overall_start
            logger.debug(
                "Answer timing (seconds): retrieval=%.3f llm=n/a total=%.3f",
                retrieval_seconds,
                total_seconds,
            )
            no_result = response_processor.build_no_result_response(
                matched_topic,
                retrieval_seconds=retrieval_seconds,
                total_seconds=total_seconds,
                no_match_message=self._no_match_message,
            )
            return self._finalize(no_result, question, suggest_rephrase, used_conversation_context)

        context = context_builder.build(chunks)
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

        llm_start = time.perf_counter()
        raw_answer_text = self._llm_client.generate_answer(
            self._system_prompt, user_prompt, conversation_history=conversation_history
        )
        llm_seconds = time.perf_counter() - llm_start
        total_seconds = time.perf_counter() - overall_start

        logger.debug(
            "Answer timing (seconds): retrieval=%.3f llm=%.3f total=%.3f",
            retrieval_seconds,
            llm_seconds,
            total_seconds,
        )
        result = response_processor.process(
            raw_answer_text,
            chunks,
            matched_topic,
            retrieval_seconds=retrieval_seconds,
            llm_seconds=llm_seconds,
            total_seconds=total_seconds,
            no_match_message=self._no_match_message,
        )
        return self._finalize(result, question, suggest_rephrase, used_conversation_context)
