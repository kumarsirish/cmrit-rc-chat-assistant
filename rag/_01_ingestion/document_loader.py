"""Step 1: Document loading.

Finds the college policy PDFs in the configured source directory and
extracts their raw text, one entry per page, via LangChain's
PyPDFLoader. Later steps (chunking, embedding) build on top of the
PageDocument objects returned here.

Basic validation lives here too, since it's the same stage that would
otherwise let a bad document in silently: a PDF that can't be opened
raises DocumentLoadError immediately (see load_pdf); load_all_documents
logs and skips a single bad file rather than aborting the whole
ingestion run, but raises DocumentLoadError itself if that leaves
nothing loaded at all.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

# A page with fewer extracted characters than this is kept (it may
# genuinely be a short page — a cover page, a single-line notice) but
# logged, since it's also the signature of a scanned/image-only page
# PyPDFLoader could only pull a caption or page number from.
MIN_EXPECTED_PAGE_CHARS = 20

# PPT-to-PDF extraction (and some PDF exports) leaves bare slide/page
# number lines interspersed with real content — a line containing
# nothing but digits (e.g. a lone "1", "20"). These carry no
# information, but sitting right next to a real definition (e.g.
# "Touch Your TOES (TYT)\n...\n1\n\n...") they've been observed to make
# the downstream LLM treat that whole source as too noisy/unclear to
# answer from and refuse outright, even though the actual answer is
# right there (see the "what does tyt stand for" investigation).
# Stripped for every PDF, not just pptx-derived ones, since a bare
# numeric line is never meaningful prose either way.
_STANDALONE_NUMBER_LINE_PATTERN = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.MULTILINE)


def _strip_standalone_number_lines(text: str) -> str:
    return _STANDALONE_NUMBER_LINE_PATTERN.sub("", text)


class DocumentLoadError(Exception):
    """Raised when a PDF cannot be opened/parsed, or yields no usable
    text at all."""


@dataclass
class PageDocument:
    """Raw text extracted from a single PDF page."""

    text: str
    source_file: str
    page_number: int


def find_pdf_files(source_directory: str) -> list:
    """Returns a sorted list of PDF file paths in source_directory."""
    directory = Path(source_directory)
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    pdf_paths = sorted(directory.glob("*.pdf"))
    return pdf_paths


def load_pdf(pdf_path: Path) -> list:
    """Extracts text from every page of a single PDF file.

    Returns a list of PageDocument, one per non-empty page. Page
    numbers are 1-indexed to match what a human would see in a PDF
    viewer, which makes citations easier to verify against the source
    (PyPDFLoader's "page" metadata is 0-indexed).

    Raises DocumentLoadError if the file can't be opened/parsed at all,
    or if every page turns out empty (a scanned/image-only PDF with no
    extractable text layer) — either way, nothing usable came out of
    this file.

    extraction_mode="layout" (rather than pypdf's default "plain")
    uses glyph positions to reconstruct lines instead of a
    cursor-jump heuristic. Several of policies_docs's PDFs are
    DOCX/PPTX exports with justified or widely-spaced text, which the
    default mode was observed splitting into one word per line (e.g.
    "Quality Policy: To deliver quality technical education to
    inculcate ... social\n\ncommitment\n\nin\n\nour\n\nstudents..."),
    fragmented enough that the LLM sometimes couldn't recognize the
    text as an answer even though it was the right chunk (see the
    "what is CMRIT's quality policy" investigation). "layout" mode was
    checked against this file plus three known-good ones (including a
    table-shaped PDF, where column alignment survived) before
    switching — see PR discussion, not just this file's history.
    """
    try:
        documents = PyPDFLoader(str(pdf_path), extraction_mode="layout").load()
    except Exception as exc:
        raise DocumentLoadError(f"Could not open/parse PDF {pdf_path.name}: {exc}") from exc

    pages = []
    for document in documents:
        page_text = _strip_standalone_number_lines(document.page_content).strip()
        if not page_text:
            continue
        if len(page_text) < MIN_EXPECTED_PAGE_CHARS:
            logger.warning(
                "%s page %d extracted only %d character(s) of text — "
                "possibly a scanned/image page with no text layer.",
                pdf_path.name,
                document.metadata["page"] + 1,
                len(page_text),
            )
        pages.append(
            PageDocument(
                text=page_text,
                source_file=pdf_path.name,
                page_number=document.metadata["page"] + 1,
            )
        )

    if not pages:
        raise DocumentLoadError(
            f"{pdf_path.name} yielded no extractable text on any page "
            "(likely scanned/image-only, or empty)."
        )
    return pages


def load_all_documents(source_directory: str) -> list:
    """Loads every PDF in source_directory into a flat list of
    PageDocument.

    A single PDF that fails validation (see load_pdf) is logged and
    skipped, so one bad document doesn't block ingesting the rest —
    but if that leaves nothing loaded at all, DocumentLoadError is
    raised, since an ingestion run with zero pages is never useful.
    """
    all_pages = []
    for pdf_path in find_pdf_files(source_directory):
        try:
            all_pages.extend(load_pdf(pdf_path))
        except DocumentLoadError as exc:
            logger.error("Skipping %s: %s", pdf_path.name, exc)

    if not all_pages:
        raise DocumentLoadError(f"No usable PDF pages found in {source_directory}.")
    return all_pages
