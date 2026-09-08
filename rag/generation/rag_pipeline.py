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
from rag.indexing.docs_metadata import find_matching_topic
from rag.response import response_processor
from rag.response.response_processor import NO_MATCH_MESSAGE, RagAnswer, SourceReference  # re-exported
from rag.retrieval.retriever import Retriever

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about CMRIT college "
    "policies. You must answer using ONLY the CONTEXT provided below, "
    "which is extracted from official policy documents. "
    "Do not use any outside knowledge, even if you believe it to be true. "
    "If the CONTEXT does not contain the answer, reply exactly: "
    f'"{NO_MATCH_MESSAGE}" '
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
    "current question."
)

logger = logging.getLogger(__name__)


class RagPipeline:
    """Answers a user question grounded in the retrieved policy chunks."""

    def __init__(self, retriever: Retriever, llm_client: LlmClient, doc_topics: list = None):
        self._retriever = retriever
        self._llm_client = llm_client
        self._doc_topics = doc_topics or []

    def answer(self, question: str, conversation_history: list = None) -> RagAnswer:
        """Answers question, grounded in freshly retrieved chunks.

        Retrieval runs on question alone — conversation_history (prior
        {"role", "content"} turns from the same session, if any) is
        never used to build the retrieval query. It's only passed to
        the LLM below, so it can resolve references like "it"/"that"
        in its reply; the retriever never sees it.
        """
        overall_start = time.perf_counter()
        chunks = self._retriever.retrieve(question)
        retrieval_seconds = time.perf_counter() - overall_start
        matched_topic = find_matching_topic(question, self._doc_topics)

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
