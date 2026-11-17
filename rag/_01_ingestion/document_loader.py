"""Step 1: Document loading.

Finds the documents (PDF, Word, plain text, Excel, PowerPoint) in the
configured source directory (optionally arranged into per-category
subfolders) and extracts their raw text, one entry per logical "page"
(PDF page, whole docx/txt file, Excel worksheet, or PowerPoint slide),
via format-specific loaders below. Later steps (chunking, embedding)
build on top of the PageDocument objects returned here, regardless of
which format produced them.

Basic validation lives here too, since it's the same stage that would
otherwise let a bad document in silently: a file that can't be
opened/parsed raises DocumentLoadError immediately; load_all_documents
logs and skips a single bad file rather than aborting the whole
ingestion run, but raises DocumentLoadError itself if that leaves
nothing loaded at all.
"""

import logging
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import openpyxl
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from langchain_community.document_loaders import PyPDFLoader
from pptx import Presentation

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


def list_categories(source_directory: str) -> list:
    """Subfolder names directly under source_directory, each one an
    existing category a document has already been saved into — used
    both by the Admin UI's category picker and by the category
    classifier (rag._03_routing.category_matcher) built from this same
    list. A brand-new deployment with nothing uploaded yet simply has
    none."""
    directory = Path(source_directory)
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


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


def extract_zip_archive(source_directory: str, topic: str, zip_content: bytes) -> list:
    """Extracts every supported document file from a zip archive's
    bytes into source_directory/<topic folder>, for the Admin UI's
    bulk-upload flow (uploading a .zip instead of a single document).

    Any folder structure inside the zip is flattened — only each
    entry's basename is used, same path-traversal protection as
    save_uploaded_document — and a same-named existing file is
    overwritten, so re-uploading a zip to update several documents at
    once just replaces them in place. Entries that are directories, or
    whose extension isn't in _LOADERS_BY_EXTENSION (including zip
    metadata cruft like __MACOSX/ or .DS_Store), are skipped.

    The zip itself is never written to disk — it's read entirely from
    the bytes already held in memory (the Admin UI's uploaded file),
    so there is nothing left to clean up once this returns.

    Returns the filenames actually extracted.
    """
    topic_directory = Path(source_directory) / sanitize_topic_folder_name(topic)
    topic_directory.mkdir(parents=True, exist_ok=True)

    extracted_filenames = []
    with zipfile.ZipFile(BytesIO(zip_content)) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            filename = Path(entry.filename).name
            if not filename or filename.startswith("."):
                continue
            if Path(filename).suffix.lower() not in _LOADERS_BY_EXTENSION:
                continue
            (topic_directory / filename).write_bytes(archive.read(entry))
            extracted_filenames.append(filename)
    return extracted_filenames


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


def _iter_docx_blocks(document):
    """Yields each top-level paragraph/table in the document's actual
    reading order. python-docx's own document.paragraphs/document.tables
    are separate flat lists that lose how paragraphs and tables are
    interleaved in the real document — walking the body's own XML
    children instead preserves that order."""
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, document)


def _table_text(table) -> str:
    """One line per row, cells joined with a wide gap — a plain-text
    shape chunker.py's table-line heuristic already recognizes as a
    table (see chunker.py's _TABLE_LINE_PATTERN), same as how a table
    inside a PDF reads once PyPDFLoader flattens it to text."""
    lines = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            lines.append("   ".join(cells))
    return "\n".join(lines)


def load_docx(docx_path: Path) -> list:
    """Extracts text from a Word document — both ordinary paragraphs
    and any tables (a checklist or form laid out entirely as a table,
    with no body paragraphs at all, previously yielded nothing: see the
    "Checklist for RC Submission.docx" investigation, where
    document.paragraphs was empty because every line of real content
    sat inside two tables).

    python-docx has no reliable way to recover Word's own page breaks
    (they're a rendering-time concept, not stored per-paragraph), so
    unlike PDF this returns a SINGLE PageDocument for the whole file
    (page_number=1) rather than one per page. Downstream consumers
    (chunker, citations) only need page_number to mean "the logical
    unit of this source file this text came from" — a single unit for
    the whole document is a valid, if less granular, value for that.
    """
    try:
        document = DocxDocument(str(docx_path))
        blocks = []
        for block in _iter_docx_blocks(document):
            block_text = block.text if isinstance(block, DocxParagraph) else _table_text(block)
            if block_text.strip():
                blocks.append(block_text)
    except Exception as exc:
        raise DocumentLoadError(f"Could not open/parse Word document {docx_path.name}: {exc}") from exc

    text = "\n\n".join(blocks).strip()
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


def load_pptx(pptx_path: Path) -> list:
    """Extracts text from a PowerPoint deck, one PageDocument per slide
    (page_number = the slide's 1-indexed position) — slides are the
    natural "page" unit here, same reasoning as one PageDocument per
    Excel worksheet in load_xlsx.

    Every shape with a text frame (title, body, text boxes) has its
    paragraphs joined into that slide's text; shapes with no text
    (images, plain lines/connectors) contribute nothing. Slides with no
    extractable text at all (e.g. image-only) are skipped, same as a
    blank PDF page.
    """
    try:
        presentation = Presentation(str(pptx_path))
    except Exception as exc:
        raise DocumentLoadError(f"Could not open/parse PowerPoint file {pptx_path.name}: {exc}") from exc

    pages = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                paragraph_text = "".join(run.text for run in paragraph.runs).strip()
                if paragraph_text:
                    lines.append(paragraph_text)
        slide_text = "\n".join(lines).strip()
        if slide_text:
            pages.append(PageDocument(text=slide_text, source_file=pptx_path.name, page_number=slide_index))

    if not pages:
        raise DocumentLoadError(f"{pptx_path.name} yielded no extractable text on any slide.")
    return pages


_LOADERS_BY_EXTENSION = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".xlsx": load_xlsx,
    ".pptx": load_pptx,
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
