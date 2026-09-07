"""Step 1: Document retrieval.

Finds the college policy PDFs in the configured source directory and
extracts their raw text, one entry per page. Later steps (chunking,
embedding) build on top of the PageDocument objects returned here.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


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
    viewer, which makes citations easier to verify against the source.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_index, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append(
                PageDocument(
                    text=page_text,
                    source_file=pdf_path.name,
                    page_number=page_index + 1,
                )
            )
    return pages


def load_all_documents(source_directory: str) -> list:
    """Loads every PDF in source_directory into a flat list of PageDocument."""
    all_pages = []
    for pdf_path in find_pdf_files(source_directory):
        all_pages.extend(load_pdf(pdf_path))
    return all_pages
