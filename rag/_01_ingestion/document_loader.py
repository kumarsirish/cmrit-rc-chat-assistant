"""Step 1: Document loading.

Finds the documents (PDF, Word, plain text, Excel) in the configured
source directory (optionally arranged into per-category subfolders) and
extracts their raw text, one entry per logical "page" (PDF page, whole
docx/txt file, or Excel worksheet), via format-specific loaders below.
Later steps (chunking, embedding) build on top of the PageDocument
objects returned here, regardless of which format produced them.

Basic validation lives here too, since it's the same stage that would
otherwise let a bad document in silently: a file that can't be
opened/parsed raises DocumentLoadError immediately; load_all_documents
logs and skips a single bad file rather than aborting the whole
ingestion run, but raises DocumentLoadError itself if that leaves
nothing loaded at all.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import openpyxl
from docx import Document as DocxDocument
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


def find_document_files(source_directory: str, archived_subfolder_name: str = "_archived") -> list:
    """Returns a sorted list of supported document file paths in
    source_directory (any extension in _LOADERS_BY_EXTENSION).

    Searches recursively (not just the top level) so documents arranged
    into per-category subfolders (e.g. source_directory/"Leave
    Policy"/*.pdf) are found the same way as ones sitting directly in
    source_directory.

    Any path with a component named archived_subfolder_name is skipped
    — a document moved there (see rag._03_routing.document_registry)
    has been manually superseded by a newer upload and must stop being
    retrievable without needing to touch the vector store directly.
    """
    directory = Path(source_directory)
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    document_paths = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _LOADERS_BY_EXTENSION
        and archived_subfolder_name not in path.relative_to(directory).parts
    ]
    return sorted(document_paths)


_UNSAFE_FOLDER_CHARS_PATTERN = re.compile(r"[^\w\s\-().,&]")


def sanitize_topic_folder_name(topic: str) -> str:
    """Turns a free-text topic name into a filesystem-safe folder name:
    strips path separators and other characters that would misbehave as
    a directory name, falling back to "Uncategorized" if nothing usable
    is left. Used by save_uploaded_document so a topic entered in the
    Admin UI can be used directly as a subfolder name."""
    name = topic.strip().replace("/", "-").replace("\\", "-")
    name = _UNSAFE_FOLDER_CHARS_PATTERN.sub("", name).strip()
    return name or "Uncategorized"


def save_uploaded_document(source_directory: str, topic: str, filename: str, content: bytes) -> Path:
    """Saves an uploaded document's bytes under source_directory/<topic
    folder>, creating the topic folder if it doesn't exist yet. Used by
    the Admin UI's upload flow. Works for any supported extension — the
    bytes are written as-is, format-specific parsing only happens later
    at ingestion time.

    filename is reduced to its basename first so a crafted name (e.g.
    containing "../") can't write outside the topic folder.
    """
    topic_directory = Path(source_directory) / sanitize_topic_folder_name(topic)
    topic_directory.mkdir(parents=True, exist_ok=True)
    document_path = topic_directory / Path(filename).name
    document_path.write_bytes(content)
    return document_path


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
    cursor-jump heuristic. Several of knowledge_documents's PDFs are
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


def load_docx(docx_path: Path) -> list:
    """Extracts text from a Word document.

    python-docx has no reliable way to recover Word's own page breaks
    (they're a rendering-time concept, not stored per-paragraph), so
    unlike PDF this returns a SINGLE PageDocument for the whole file
    (page_number=1) rather than one per page. Downstream consumers
    (chunker, citations) only need page_number to mean "the logical
    unit of this source file this text came from" — a single unit for
    the whole document is a valid, if less granular, value for that.
    """
    try:
        paragraphs = [paragraph.text for paragraph in DocxDocument(str(docx_path)).paragraphs]
    except Exception as exc:
        raise DocumentLoadError(f"Could not open/parse Word document {docx_path.name}: {exc}") from exc

    text = "\n".join(paragraphs).strip()
    if not text:
        raise DocumentLoadError(f"{docx_path.name} yielded no extractable text.")
    return [PageDocument(text=text, source_file=docx_path.name, page_number=1)]


def load_txt(txt_path: Path) -> list:
    """Extracts text from a plain-text file. Single PageDocument, same
    reasoning as load_docx — a .txt file has no page concept at all."""
    try:
        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        raise DocumentLoadError(f"Could not open text file {txt_path.name}: {exc}") from exc

    if not text:
        raise DocumentLoadError(f"{txt_path.name} yielded no extractable text.")
    return [PageDocument(text=text, source_file=txt_path.name, page_number=1)]


def load_xlsx(xlsx_path: Path) -> list:
    """Extracts text from an Excel workbook, one PageDocument per
    worksheet (page_number = the sheet's 1-indexed position).

    Content here is confirmed free-text heavy (not tabular records), so
    each row's non-empty cells are simply joined into a line of prose
    rather than parsed as structured data — the same shape chunker.py
    already expects from any other text source.
    """
    try:
        workbook = openpyxl.load_workbook(str(xlsx_path), data_only=True, read_only=True)
    except Exception as exc:
        raise DocumentLoadError(f"Could not open/parse Excel workbook {xlsx_path.name}: {exc}") from exc

    pages = []
    for sheet_index, worksheet in enumerate(workbook.worksheets, start=1):
        lines = []
        for row in worksheet.iter_rows(values_only=True):
            cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
            if cells:
                lines.append(" ".join(cells))
        sheet_text = "\n".join(lines).strip()
        if sheet_text:
            pages.append(PageDocument(text=sheet_text, source_file=xlsx_path.name, page_number=sheet_index))

    if not pages:
        raise DocumentLoadError(f"{xlsx_path.name} yielded no extractable text on any sheet.")
    return pages


_LOADERS_BY_EXTENSION = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".xlsx": load_xlsx,
}


def load_all_documents(source_directory: str, archived_subfolder_name: str = "_archived") -> list:
    """Loads every supported document in source_directory into a flat
    list of PageDocument.

    A single file that fails validation (see the per-format load_*
    functions) is logged and skipped, so one bad document doesn't block
    ingesting the rest — but if that leaves nothing loaded at all,
    DocumentLoadError is raised, since an ingestion run with zero pages
    is never useful.
    """
    all_pages = []
    for document_path in find_document_files(source_directory, archived_subfolder_name):
        loader = _LOADERS_BY_EXTENSION[document_path.suffix.lower()]
        try:
            all_pages.extend(loader(document_path))
        except DocumentLoadError as exc:
            logger.error("Skipping %s: %s", document_path.name, exc)

    if not all_pages:
        raise DocumentLoadError(f"No usable document pages found in {source_directory}.")
    return all_pages
