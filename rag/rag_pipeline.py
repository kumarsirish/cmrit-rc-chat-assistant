"""Step 6: Ties retrieval and generation together.

Produces answers that are grounded in the retrieved policy text only.
Two layers keep the assistant from answering with outside knowledge:

1. If retrieval finds no chunk above the relevance threshold, the
   pipeline refuses immediately without calling the LLM at all.
2. If chunks are found, the LLM is given a strict system prompt that
   restricts it to the supplied context and told to say so when the
   context does not contain the answer.
"""

import re
from dataclasses import dataclass

from rag.llm_client import LlmClient
from rag.retriever import Retriever

NO_MATCH_MESSAGE = (
    "I could not find anything about that in the college policy documents. "
    "I can only answer questions covered by those documents."
)

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about CMRIT college "
    "policies. You must answer using ONLY the CONTEXT provided below, "
    "which is extracted from official policy documents. "
    "Do not use any outside knowledge, even if you believe it to be true. "
    "If the CONTEXT does not contain the answer, reply exactly: "
    f'"{NO_MATCH_MESSAGE}" '
    "Keep URLs from the context exactly as written if you include them. "
    "Write only the answer itself — do not mention document names, page "
    "numbers, or the words 'Source'/'CONTEXT' anywhere in your reply, and "
    "never add bracketed citations like '[Source 1: ...]'. The application "
    "displays the source documents separately below your answer, so any "
    "citation you add would just be duplicated noise. "
    "Earlier turns from this conversation may be included before the "
    "current question purely to resolve references like 'it' or 'that "
    "leave type' — they are not an additional source of facts. Every "
    "factual claim must still come from the CONTEXT given with the "
    "current question."
)

# Belt-and-braces cleanup: even with the instruction above, a model can
# still slip in a citation — bracketed ("[Source 1: policy.pdf, page 2]")
# or as plain prose ("...specifically from Source 1: policy.pdf, page
# 20..."). Strip both shapes so the answer text never duplicates what
# the UI already renders as separate source links.
_CONTEXT_LEAD_IN_PATTERN = re.compile(
    r"According to (the )?(provided )?CONTEXT,?\s*(specifically from\s*)?",
    re.IGNORECASE,
)
_INLINE_CITATION_PATTERN = re.compile(
    r"\[?\s*Source\s+\d+\s*:\s*[^,\]]+,\s*pages?\s*\d+(?:-\d+)?\s*\]?,?\s*",
    re.IGNORECASE,
)


def _strip_inline_citations(answer_text: str) -> str:
    cleaned = _CONTEXT_LEAD_IN_PATTERN.sub("", answer_text)
    cleaned = _INLINE_CITATION_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip().lstrip(",").strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


@dataclass
class SourceReference:
    source_file: str
    page_number: int
    relevance_score: float
    text_preview: str
    match_type: str


@dataclass
class RagAnswer:
    answer_text: str
    sources: list
    grounded: bool


def _build_retrieval_query(question: str, conversation_history: list) -> str:
    """Broadens the retrieval query with the previous user turn.

    A short follow-up like "give details" or "what about that" carries
    no topic words of its own, so searching for it in isolation finds
    nothing. Prepending the previous user turn gives retrieval enough
    signal to find the chunks that turn was about, without dragging in
    the whole conversation (which would blur the query for a genuinely
    new, unrelated question).
    """
    if not conversation_history:
        return question

    previous_user_message = None
    for message in reversed(conversation_history):
        if message["role"] == "user":
            previous_user_message = message["content"]
            break

    if not previous_user_message:
        return question
    return f"{previous_user_message} {question}"


def _build_context(chunks: list) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {index}: {chunk.source_file}, page {chunk.page_number}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(context_parts)


class RagPipeline:
    """Answers a user question grounded in the retrieved policy chunks."""

    def __init__(self, retriever: Retriever, llm_client: LlmClient):
        self._retriever = retriever
        self._llm_client = llm_client

    def answer(self, question: str, conversation_history: list = None) -> RagAnswer:
        """Answers question, grounded in freshly retrieved chunks.

        conversation_history (optional) is prior turns from the same
        session — {"role": "user"/"assistant", "content": str} dicts.
        It's used two ways: retrieval is broadened with the previous
        user turn so a topic-less follow-up ("give details") can still
        find the right chunks (see _build_retrieval_query), and the
        full history is passed to the LLM so it can resolve references
        like "it"/"that" in its reply.
        """
        retrieval_query = _build_retrieval_query(question, conversation_history)
        chunks = self._retriever.retrieve(retrieval_query)

        if not chunks:
            return RagAnswer(answer_text=NO_MATCH_MESSAGE, sources=[], grounded=False)

        context = _build_context(chunks)
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        raw_answer_text = self._llm_client.generate_answer(
            SYSTEM_PROMPT, user_prompt, conversation_history=conversation_history
        )
        answer_text = _strip_inline_citations(raw_answer_text)

        sources = [
            SourceReference(
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                relevance_score=chunk.relevance_score,
                text_preview=chunk.text[:200],
                match_type=chunk.match_type,
            )
            for chunk in chunks
        ]
        grounded = NO_MATCH_MESSAGE not in answer_text
        return RagAnswer(answer_text=answer_text, sources=sources, grounded=grounded)
