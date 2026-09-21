"""Step 4 (context construction): formats retrieved chunks into the
CONTEXT block sent to the LLM.

Only responsible for formatting — it does not retrieve chunks or call
the LLM. Each chunk becomes a numbered "Source N" block (1-based, in
the order the retriever returned them); rag_pipeline's citation
handling relies on that numbering matching the "[N]" markers the model
is asked to cite with.
"""


def _page_label(chunk) -> str:
    if chunk.page_end and chunk.page_end != chunk.page_number:
        return f"pages {chunk.page_number}-{chunk.page_end}"
    return f"page {chunk.page_number}"


def build(chunks: list) -> str:
    """Returns the CONTEXT string for chunks, one numbered block per
    chunk in the given order, each carrying its source file, page(s),
    and section title (when detected) so the LLM can cite it."""
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        section_note = f", section: \"{chunk.section_title}\"" if chunk.section_title else ""
        confidence_note = " — LOW-CONFIDENCE MATCH" if chunk.low_confidence else ""
        context_parts.append(
            f"[Source {index}: {chunk.source_file}, {_page_label(chunk)}{section_note}{confidence_note}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(context_parts)
