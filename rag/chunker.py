"""Step 2: Chunking.

Splits page text into overlapping chunks using a recursive strategy:
try to split on paragraph breaks first, then line breaks, then
sentence stops, and only fall back to a hard character cut if a piece
is still too large. This keeps related sentences together instead of
cutting text at an arbitrary character offset.

URLs are masked before splitting on sentence stops (". ") so that a
period inside a URL (e.g. "https://cmrit.ac.in/policy.pdf") is never
mistaken for a sentence boundary and does not get split apart.
"""

import re
from dataclasses import dataclass

URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")

# PDF text extraction often leaves a stray space/tab on an otherwise
# blank line (e.g. "...end of section.\n \nNext heading..."), so the
# line isn't literally "\n\n" and the paragraph separator below never
# matches it. Left alone, this lets unrelated sections get merged into
# the same chunk across what should be a hard paragraph break.
_BLANK_LINE_PATTERN = re.compile(r"\n[ \t]+\n")


def _normalize_blank_lines(text: str) -> str:
    return _BLANK_LINE_PATTERN.sub("\n\n", text)


@dataclass
class Chunk:
    """A chunk of policy text ready to be embedded and stored."""

    text: str
    source_file: str
    page_number: int
    chunk_index: int


def _mask_urls(text: str) -> tuple:
    """Replaces URLs with placeholder tokens that contain no periods.

    Returns the masked text plus a mapping from placeholder to the
    original URL, so the URLs can be restored after splitting.
    """
    urls_found = URL_PATTERN.findall(text)
    masked_text = text
    placeholder_map = {}
    for index, url in enumerate(urls_found):
        placeholder = f"__URL_PLACEHOLDER_{index}__"
        placeholder_map[placeholder] = url
        masked_text = masked_text.replace(url, placeholder, 1)
    return masked_text, placeholder_map


def _unmask_urls(text: str, placeholder_map: dict) -> str:
    restored_text = text
    for placeholder, url in placeholder_map.items():
        restored_text = restored_text.replace(placeholder, url)
    return restored_text


def _split_on_separator(text: str, separator: str) -> list:
    if separator == "":
        return list(text)
    pieces = text.split(separator)
    # Re-attach the separator so downstream joins reproduce the
    # original text faithfully (important for ". " sentence stops).
    rejoined = []
    for index, piece in enumerate(pieces):
        if index < len(pieces) - 1:
            rejoined.append(piece + separator)
        elif piece:
            rejoined.append(piece)
    return rejoined


def _recursive_split(text: str, separators: list, chunk_size: int) -> list:
    """Recursively splits text using the first separator that yields
    pieces small enough to fit in chunk_size, falling back to the
    next separator (and eventually raw characters) when needed.
    """
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # No separators left: hard-cut by character count.
        return [
            text[start : start + chunk_size]
            for start in range(0, len(text), chunk_size)
        ]

    separator, remaining_separators = separators[0], separators[1:]
    pieces = _split_on_separator(text, separator)

    result = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(_recursive_split(piece, remaining_separators, chunk_size))
    return result


def _merge_pieces(pieces: list, chunk_size: int, chunk_overlap: int) -> list:
    """Greedily combines small pieces into chunks close to chunk_size,
    carrying a small overlap from the end of one chunk into the next
    so context is not lost at chunk boundaries.
    """
    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = overlap_text + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, separators: list) -> list:
    """Splits a single block of text into overlapping chunks, preserving URLs."""
    text = _normalize_blank_lines(text)
    masked_text, placeholder_map = _mask_urls(text)
    pieces = _recursive_split(masked_text, separators, chunk_size)
    masked_chunks = _merge_pieces(pieces, chunk_size, chunk_overlap)
    return [_unmask_urls(chunk, placeholder_map) for chunk in masked_chunks]


def chunk_documents(
    pages: list, chunk_size: int, chunk_overlap: int, separators: list
) -> list:
    """Chunks every PageDocument and returns a flat list of Chunk objects."""
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page.text, chunk_size, chunk_overlap, separators)
        for chunk_index, chunk_text_value in enumerate(page_chunks):
            all_chunks.append(
                Chunk(
                    text=chunk_text_value,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )
    return all_chunks
