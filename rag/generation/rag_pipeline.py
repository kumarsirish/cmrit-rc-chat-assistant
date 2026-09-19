"""Step 6: Ties retrieval and generation together.

Produces answers that are grounded in the retrieved policy text only.
Two layers keep the assistant from answering with outside knowledge:

1. If retrieval finds no chunk above the relevance threshold, the
   pipeline refuses immediately without calling the LLM at all.
2. If chunks are found, the LLM is given a strict system prompt that
   restricts it to the supplied context and told to say so when the
   context does not contain the answer.

RagPipeline itself only orchestrates: it delegates context formatting
to context_builder and citation/response assembly to
response_processor, so each of those concerns can be read, tested, and
changed independently of the others. Retrieval always runs on the
current question alone — conversation history is never used to build
the retrieval query, only passed to the LLM (see answer() below) so it
can resolve references like "it" in its reply. This keeps retrieval
deterministic, fast, and easy to evaluate/debug independent of session
state.
"""

import logging
import time

from rag.generation import context_builder
from rag.generation.llm_client import LlmClient
from rag.response import response_processor
from rag.response.response_processor import NO_MATCH_MESSAGE, RagAnswer, SourceReference  # re-exported
from rag.retrieval.retrieval_service import RetrievalService

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about CMRIT college "
    "policies. You must answer using ONLY the CONTEXT provided below, "
    "which is extracted from official policy documents. "
    "Do not use any outside knowledge, even if you believe it to be true. "
    "The CONTEXT sources vary in quality — some are extracted from "
    "slide decks or other messy layouts and may contain stray "
    "fragments, page numbers, or irrelevant text alongside the useful "
    "part. Judge each source independently: if even one source clearly "
    "supports an answer, use it and cite it, even if other sources in "
    "the CONTEXT are noisy, poorly formatted, or unrelated to the "
    "question. Never let noise or irrelevance in one source stop you "
    "from using a clear answer found in another. "
    "If the CONTEXT does not contain the answer, reply exactly: "
    f'"{NO_MATCH_MESSAGE}" '
    "Answer ONLY the specific question asked — nothing more. Do not "
    "add related rules, exceptions, background, or nearby policies "
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
    "bullet points instead of prose."
)

logger = logging.getLogger(__name__)


class RagPipeline:
    """Answers a user question grounded in the retrieved policy chunks."""

    def __init__(self, retrieval_service: RetrievalService, llm_client: LlmClient):
        self._retrieval_service = retrieval_service
        self._llm_client = llm_client

    def answer(
        self,
        question: str,
        conversation_history: list = None,
        top_k: int = None,
        min_reranker_score: float = None,
    ) -> RagAnswer:
        """Answers question, grounded in freshly retrieved chunks.

        Retrieval runs on question alone — conversation_history (prior
        {"role", "content"} turns from the same session, if any) is
        never used to build the retrieval query. It's only passed to
        the LLM below, so it can resolve references like "it"/"that"
        in its reply; retrieval never sees it.

        top_k / min_reranker_score, if given, override the retrieval
        service's configured defaults for this call only (used by
        evaluation scripts to try settings without editing config.yaml).

        retrieval_service.retrieve() returns the matched topic along
        with the chunks, so it only has to detect/embed the topic
        match once per question instead of a second time here.
        """
        overall_start = time.perf_counter()
        chunks, matched_topic = self._retrieval_service.retrieve(
            question, top_k=top_k, min_reranker_score=min_reranker_score
        )
        retrieval_seconds = time.perf_counter() - overall_start

        if not chunks:
            total_seconds = time.perf_counter() - overall_start
            logger.debug(
                "Answer timing (seconds): retrieval=%.3f llm=n/a total=%.3f",
                retrieval_seconds,
                total_seconds,
            )
            return response_processor.build_no_result_response(
                matched_topic, retrieval_seconds=retrieval_seconds, total_seconds=total_seconds
            )

        context = context_builder.build(chunks)
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

        llm_start = time.perf_counter()
        raw_answer_text = self._llm_client.generate_answer(
            SYSTEM_PROMPT, user_prompt, conversation_history=conversation_history
        )
        llm_seconds = time.perf_counter() - llm_start
        total_seconds = time.perf_counter() - overall_start

        logger.debug(
            "Answer timing (seconds): retrieval=%.3f llm=%.3f total=%.3f",
            retrieval_seconds,
            llm_seconds,
            total_seconds,
        )
        return response_processor.process(
            raw_answer_text,
            chunks,
            matched_topic,
            retrieval_seconds=retrieval_seconds,
            llm_seconds=llm_seconds,
            total_seconds=total_seconds,
        )
