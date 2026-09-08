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
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

# A page with fewer extracted characters than this is kept (it may
# genuinely be a short page — a cover page, a single-line notice) but
# logged, since it's also the signature of a scanned/image-only page
# PyPDFLoader could only pull a caption or page number from.
MIN_EXPECTED_PAGE_CHARS = 20


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
    """
    try:
        documents = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:
        raise DocumentLoadError(f"Could not open/parse PDF {pdf_path.name}: {exc}") from exc

    pages = []
    for document in documents:
        page_text = document.page_content.strip()
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
