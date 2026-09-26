"""Step 6 (response processing): turns a raw LLM reply + retrieved
chunks into the final RagAnswer the UI renders.

Independent of retrieval and generation: it only looks at the chunks
it's given and the text it's given, never calls the retriever or the
LLM itself. Three things happen here:

1. Citation validation — the model is asked (see rag_pipeline.
   SYSTEM_PROMPT) to cite claims as "[N]", N being a "Source N" number
   from the context. A model can still slip into prose citations or
   invent a number with nothing behind it; _normalize_citations
   rewrites the former and drops the latter so every "[N]" left in the
   answer is real.
2. Source mapping — one SourceReference per retrieved chunk, carrying
   exactly the metadata already produced by chunking/ingestion
   (source file, page range, section, link) — nothing here fabricates
   or re-derives that metadata.
3. No-answer handling — build_no_result_response covers both "nothing
   was retrieved at all" and "nothing was retrieved, but the question
   matched a known topic with no locally-indexed document for it".
"""

import re
from dataclasses import dataclass

NO_MATCH_MESSAGE = (
    "I could not find any information about this query in the college policy documents."
)

# Belt-and-braces cleanup for a model that reverts to prose citations
# instead of the requested "[N]" form ("...specifically from Source 1:
# policy.pdf, page 20...", or "[Source 1: policy.pdf, page 2]") —
# normalize both into the short form so the answer's only citation
# shape is "[N]".
_CONTEXT_LEAD_IN_PATTERN = re.compile(
    r"According to (the )?(provided )?CONTEXT,?\s*(specifically from\s*)?",
    re.IGNORECASE,
)
_VERBOSE_CITATION_PATTERN = re.compile(
    r"\s*\[?\s*Source\s+(\d+)\s*(?::\s*[^,\]]+,\s*pages?\s*\d+(?:-\d+)?"
    r"(?:,\s*section\s*:\s*\"[^\"]*\")?)?\s*\]?,?\s*",
    re.IGNORECASE,
)
_CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


def _normalize_citations(answer_text: str, num_sources: int) -> str:
    """Rewrites any verbose "Source N: ..." citation into the short
    "[N]" form, then drops any "[N]" whose N doesn't correspond to a
    real, retrieved source — the model can otherwise fabricate a
    citation number that looks plausible but points at nothing.
    """
    cleaned = _CONTEXT_LEAD_IN_PATTERN.sub("", answer_text)
    cleaned = _VERBOSE_CITATION_PATTERN.sub(lambda match: f"[{match.group(1)}]", cleaned)

    def _keep_if_valid(match: re.Match) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= num_sources else ""

    cleaned = _CITATION_MARKER_PATTERN.sub(_keep_if_valid, cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)  # dangling space left by a stripped citation
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip().lstrip(",").strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _cited_source_indices(answer_text: str, num_sources: int) -> list:
    """Returns the (1-based, ascending, deduplicated) source numbers
    actually cited in answer_text, after normalization — i.e. only
    numbers that correspond to a real retrieved chunk."""
    indices = {int(match) for match in _CITATION_MARKER_PATTERN.findall(answer_text)}
    return sorted(index for index in indices if 1 <= index <= num_sources)


@dataclass
class SourceReference:
    """One retrieved chunk as surfaced to the UI — the same source
    metadata the retriever produced, carried through unchanged."""

    source_file: str
    page_number: int
    relevance_score: float
    text_preview: str
    match_type: str
    source_link: str = ""
    page_end: int = None
    section_title: str = ""
    # True when this chunk is the retrieval_service low-confidence
    # fallback (nothing cleared min_reranker_score, so the single
    # best-scoring candidate was returned anyway) — see
    # RetrievedChunk.low_confidence.
    low_confidence: bool = False

    def __post_init__(self):
        if self.page_end is None:
            self.page_end = self.page_number


@dataclass
class RagAnswer:
    """The pipeline's final result — everything the UI needs to render
    one turn, and nothing it needs to compute itself."""

    answer_text: str
    sources: list
    grounded: bool
    topic_name: str = None
    topic_link: str = None
    # 1-based indices into `sources` that answer_text actually cites
    # inline as "[N]", after stripping any fabricated/out-of-range
    # number the model might have produced — see _cited_source_indices.
    cited_indices: list = None
    # How long this turn took, in seconds — for the UI to surface
    # average response time. llm_seconds is None when the LLM was
    # never called (no chunks found).
    retrieval_seconds: float = None
    llm_seconds: float = None
    total_seconds: float = None
    # True when any source is the low-confidence fallback (see
    # SourceReference.low_confidence) — callers (dashboard.py) should
    # show a clear warning telling the user to verify against the
    # source directly, since this answer isn't grounded in a chunk
    # that confidently matched the question.
    low_confidence: bool = False
    # Set only when grounded is False — an alternative phrasing of the
    # question the UI can offer the user, see
    # rag._05_generation.rephrase_suggester. None whenever grounded is
    # True, or when generating a suggestion itself failed.
    rephrase_suggestion: str = None

    def __post_init__(self):
        if self.cited_indices is None:
            self.cited_indices = []


def build_no_result_response(
    matched_topic, retrieval_seconds: float = None, total_seconds: float = None
) -> RagAnswer:
    """Builds the response for "nothing was retrieved".

    matched_topic (see topic_matcher.EmbeddingTopicMatcher), if set,
    means the question was recognized as being about a known topic
    that simply has no locally-indexed document — the reply points at
    that topic's official link instead of a bare refusal.
    """
    if matched_topic:
        answer_text = (
            f"I don't have \"{matched_topic.topic}\" indexed as searchable "
            "text, but here is the official resource for it:"
        )
        return RagAnswer(
            answer_text=answer_text,
            sources=[],
            grounded=False,
            topic_name=matched_topic.topic,
            topic_link=matched_topic.link,
            retrieval_seconds=retrieval_seconds,
            total_seconds=total_seconds,
        )
    return RagAnswer(
        answer_text=NO_MATCH_MESSAGE,
        sources=[],
        grounded=False,
        retrieval_seconds=retrieval_seconds,
        total_seconds=total_seconds,
    )


def process(
    raw_answer_text: str,
    chunks: list,
    matched_topic,
    retrieval_seconds: float = None,
    llm_seconds: float = None,
    total_seconds: float = None,
) -> RagAnswer:
    """Builds the final RagAnswer for a normal (chunks-found) turn:
    validates/normalizes the LLM's inline citations against the real
    retrieved chunks, maps each chunk to a SourceReference, and decides
    whether the answer counts as grounded.
    """
    answer_text = _normalize_citations(raw_answer_text, num_sources=len(chunks))
    cited_indices = _cited_source_indices(answer_text, num_sources=len(chunks))
    grounded = NO_MATCH_MESSAGE not in answer_text

    # When the LLM decided the retrieved chunks don't actually answer
    # the question (grounded=False), those chunks were context that
    # got rejected, not sources the answer relies on — showing them
    # would wrongly imply the refusal came from those documents.
    sources = (
        [
            SourceReference(
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                relevance_score=chunk.relevance_score,
                text_preview=chunk.text[:200],
                match_type=chunk.match_type,
                source_link=chunk.source_link,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                low_confidence=chunk.low_confidence,
            )
            for chunk in chunks
        ]
        if grounded
        else []
    )
    return RagAnswer(
        answer_text=answer_text,
        sources=sources,
        grounded=grounded,
        topic_name=matched_topic.topic if matched_topic else None,
        topic_link=matched_topic.link if matched_topic else None,
        cited_indices=cited_indices if grounded else [],
        retrieval_seconds=retrieval_seconds,
        llm_seconds=llm_seconds,
        total_seconds=total_seconds,
        low_confidence=grounded and any(source.low_confidence for source in sources),
    )
