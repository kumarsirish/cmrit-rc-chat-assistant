"""Step 2: Chunking.

Splits a document's text into overlapping chunks with LangChain's
RecursiveCharacterTextSplitter: try to split on detected section
headings first, then paragraph breaks, then line breaks, then sentence
stops, and only fall back to a hard character cut if a piece is still
too large. This keeps related sentences together instead of cutting
text at an arbitrary character offset.

Chunking runs over a whole document's text at once (all of its pages
joined together), not one page at a time, so a sentence, table, or
policy section that happens to straddle a page break is no longer cut
in half between two unrelated chunks. Each chunk still carries the
page(s) it came from (page_number/page_end) and, where one could be
detected, the section heading it falls under (section_title).

Heading and table detection are heuristic, regex-based pattern
matches over the plain extracted text — PyPDFLoader's extraction
carries no font size/boldness/layout info, so this can't recognize a
heading or table the way a human looking at the PDF would. It reduces
mid-section and mid-table splitting; it doesn't guarantee it.

URLs are masked before splitting on sentence stops (". ") so that a
period inside a URL (e.g. "https://cmrit.ac.in/policy.pdf") is never
mistaken for a sentence boundary and does not get split apart.
"""

import itertools
import logging
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")

# PDF text extraction often leaves a stray space/tab on an otherwise
# blank line (e.g. "...end of section.\n \nNext heading..."), so the
# line isn't literally "\n\n" and the paragraph separator below never
# matches it. Left alone, this lets unrelated sections get merged into
# the same chunk across what should be a hard paragraph break.
_BLANK_LINE_PATTERN = re.compile(r"\n[ \t]+\n")

# A line is treated as a probable section heading if it starts with
# one of these prefixes ("1. ", "1.2 ", "Section 3", "Annexure A",
# "Chapter 4") ...
_NUMBERED_HEADING_PREFIX_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s|[A-Za-z]\.\s|Section\s+\d+|Annexure\s+[A-Za-z0-9]+|Chapter\s+\d+)",
    re.IGNORECASE,
)
# ... or doesn't end in sentence punctuation and is either mostly
# uppercase ("MANDATORY TRAINING FOR NEW FACULTY") or short Title Case
# ("Roles And Responsibilities") — the shape of a heading, not a
# sentence.
_HEADING_TRAILING_PUNCTUATION_PATTERN = re.compile(r"[.,;]\s*$")
_MAX_HEADING_LENGTH = 80
_MAX_HEADING_WORDS = 10

# Nesting depth of a numbered heading: "1. " is level 1, "1.2 " is
# level 2, "1.2.3 " is level 3. Headings with no numeric prefix
# (all-caps or Title Case) are always level 1 — there is no reliable
# way to tell "MANDATORY TRAINING" apart from a subsection by text
# shape alone, so it resets the hierarchy back to the top.
_NUMERIC_HEADING_LEVEL_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")

HEADING_MARKER = "\x00HEADING_BREAK\x00"
PAGE_JOIN_SEPARATOR = "\n\n"

# A line with a run of 3+ spaces between non-space text looks like a
# table row (column-aligned plain-text extraction of a table). Two or
# more such lines in a row are masked as a single opaque block before
# splitting, so the splitter can't carve a table apart unless the
# whole table alone exceeds chunk_size.
#
# Deliberately 3+, not 2+: PDF extraction from Word/Google Docs
# exports (the *.docx.pdf files in this corpus) commonly doubles every
# single space between words as a justification artifact, so a 2+
# threshold makes ordinary prose paragraphs match on nearly every
# line. That falsely classified whole multi-page documents as "one
# giant table," bypassing chunk_size entirely and producing 4000-9000
# character chunks the splitter could never break apart — burying real
# answers (e.g. a contact-list entry) inside a chunk so dominated by
# unrelated content that retrieval couldn't find it. Genuine table
# columns are visually spaced wider (3+ spaces) to stay readable, so
# this still catches real tables (see test_table_like_block_is_not_split_mid_row)
# without swallowing doubled-space prose.
_TABLE_LINE_PATTERN = re.compile(r"\S {3,}\S")
_MIN_TABLE_BLOCK_LINES = 2

# Google Docs/Word exports of a hyperlinked table-of-contents row
# repeat the link's display text in the plain-text extraction: either
# twice back-to-back on the same line (cell text immediately followed
# by anchor text, e.g. "3  Roles  and  Responsibilities  Roles  and
# Responsibilities"), or on its own short line shortly after the row's
# title was already stated (title line, then a URL line, then the
# link's anchor text repeated alone, e.g. "5  Faculty  Handbook
# (...)" / "https://..." / "Faculty  Handbook"). No genuine policy
# sentence repeats a multi-word phrase like this, so it reliably flags
# a link-index row rather than prose.
_DUPLICATED_LINK_ROW_PATTERN = re.compile(r"^\d*\s*(.{3,80}?)\s+\1\s*$")
_MAX_REPEATED_TITLE_WORDS = 6
_REPEATED_TITLE_LOOKBACK = 2
# A chunk this dominated by such rows is a table of contents pointing
# at other documents, not answerable content — but it repeats topic
# words densely enough to outscore the real answer in BM25/reranking
# (see test_chunk_documents_drops_link_index_chunks for the regression
# this guards against), so it's dropped rather than indexed.
_MIN_LINK_INDEX_ROWS = 2
_MIN_LINK_INDEX_ROW_RATIO = 0.3


def _looks_like_link_index(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    normalized = [re.sub(r"\s+", " ", line) for line in lines]

    matches = 0
    for index, line in enumerate(normalized):
        if _DUPLICATED_LINK_ROW_PATTERN.match(line):
            matches += 1
            continue
        if len(line.split()) <= _MAX_REPEATED_TITLE_WORDS and any(
            line.lower() in earlier.lower()
            for earlier in normalized[max(0, index - _REPEATED_TITLE_LOOKBACK) : index]
        ):
            matches += 1

    return matches >= _MIN_LINK_INDEX_ROWS and matches / len(lines) >= _MIN_LINK_INDEX_ROW_RATIO


def _normalize_blank_lines(text: str) -> str:
    return _BLANK_LINE_PATTERN.sub("\n\n", text)


@dataclass
class Chunk:
    """A chunk of policy text ready to be embedded and stored."""

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    page_end: int = None
    section_title: str = ""

    def __post_init__(self):
        if self.page_end is None:
            self.page_end = self.page_number


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


def _mask_tables(text: str) -> tuple:
    """Replaces runs of table-like lines with a single placeholder
    token each, so the splitter treats a whole table as one
    indivisible unit instead of many short, unrelated-looking lines.

    Returns the masked text plus a mapping from placeholder to the
    original block of lines, so tables can be restored after
    splitting.
    """
    lines = text.split("\n")
    result_lines = []
    placeholder_map = {}
    index = 0
    table_count = 0
    while index < len(lines):
        if not _TABLE_LINE_PATTERN.search(lines[index]):
            result_lines.append(lines[index])
            index += 1
            continue

        start = index
        while index < len(lines) and (
            _TABLE_LINE_PATTERN.search(lines[index]) or lines[index].strip() == ""
        ):
            index += 1
        end = index
        while end > start and lines[end - 1].strip() == "":
            end -= 1

        block_lines = lines[start:end]
        table_line_count = sum(1 for line in block_lines if _TABLE_LINE_PATTERN.search(line))
        if table_line_count >= _MIN_TABLE_BLOCK_LINES:
            placeholder = f"__TABLE_PLACEHOLDER_{table_count}__"
            placeholder_map[placeholder] = "\n".join(block_lines)
            result_lines.append(placeholder)
            table_count += 1
        else:
            result_lines.extend(block_lines)
        index = end
    return "\n".join(result_lines), placeholder_map


def _unmask_tables(text: str, placeholder_map: dict) -> str:
    restored_text = text
    for placeholder, block in placeholder_map.items():
        restored_text = restored_text.replace(placeholder, block)
    return restored_text


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > _MAX_HEADING_LENGTH:
        return False
    if _HEADING_TRAILING_PUNCTUATION_PATTERN.search(line):
        return False
    if _NUMBERED_HEADING_PREFIX_PATTERN.match(line):
        return True

    letters = [character for character in line if character.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(1 for character in letters if character.isupper()) / len(letters)
    if uppercase_ratio > 0.8:
        return True

    words = line.split()
    if 1 <= len(words) <= _MAX_HEADING_WORDS and all(
        word[0].isupper() for word in words if word[0].isalpha()
    ):
        return True
    return False


def _heading_level(heading_text: str) -> int:
    match = _NUMERIC_HEADING_LEVEL_PATTERN.match(heading_text)
    if not match:
        return 1
    return match.group(1).count(".") + 1


def _detect_headings(text: str) -> list:
    """Scans text line by line for heading-like lines that sit alone,
    surrounded by blank lines (or the start/end of the text).

    Returns [(offset, heading_text, level), ...] in text's own
    coordinates, in ascending offset order. level is the heading's
    nesting depth (see _heading_level) and is used to build a
    "Section > Subsection" path for chunks that fall under nested
    headings, rather than just the single nearest one.
    """
    headings = []
    offset = 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and _looks_like_heading(stripped):
            previous_blank = index == 0 or lines[index - 1].strip() == ""
            next_blank = index == len(lines) - 1 or lines[index + 1].strip() == ""
            if previous_blank and next_blank:
                headings.append(
                    (offset + line.index(stripped), stripped, _heading_level(stripped))
                )
        offset += len(line) + 1  # +1 accounts for the '\n' split on
    return headings


def _insert_heading_markers(text: str, headings: list) -> str:
    """Inserts HEADING_MARKER right before each detected heading's
    offset in text, so the splitter can use it as a hard split point.
    """
    if not headings:
        return text
    parts = []
    cursor = 0
    for offset, _, _ in headings:
        parts.append(text[cursor:offset])
        parts.append(HEADING_MARKER)
        cursor = offset
    parts.append(text[cursor:])
    return "".join(parts)


def _heading_paths(headings: list) -> list:
    """Turns the flat, ascending-offset heading list into
    [(offset, path), ...], where path is a " > "-joined chain from the
    top-level heading active at that point down to the heading itself
    (e.g. "1. Leave Policy > 1.2 Casual Leave").

    A heading whose level is <= an ancestor still on the stack closes
    that ancestor (and anything nested under it) first, so the stack
    always reflects genuine nesting rather than every heading seen
    so far.
    """
    paths = []
    stack = []  # [(level, text), ...], outermost first
    for offset, text, level in headings:
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        paths.append((offset, " > ".join(heading_text for _, heading_text in stack)))
    return paths


def _section_title_for_offset(heading_paths: list, offset: int) -> str:
    """Returns the section/subsection path active at or before offset,
    or "" if no heading precedes it."""
    section_title = ""
    for heading_offset, path in heading_paths:
        if heading_offset > offset:
            break
        section_title = path
    return section_title


def _join_pages(pages: list) -> tuple:
    """Joins a document's pages into one continuous string.

    Returns (full_text, page_spans), where page_spans is a list of
    (start_offset, end_offset, page_number) tuples in full_text's own
    coordinates, in page order.
    """
    text_parts = []
    page_spans = []
    offset = 0
    for index, page in enumerate(pages):
        start = offset
        text_parts.append(page.text)
        offset += len(page.text)
        page_spans.append((start, offset, page.page_number))
        if index < len(pages) - 1:
            offset += len(PAGE_JOIN_SEPARATOR)
    return PAGE_JOIN_SEPARATOR.join(text_parts), page_spans


def _pages_for_span(page_spans: list, start: int, end: int) -> tuple:
    """Returns (start_page, end_page) covering the [start, end) span."""
    start_page = page_spans[0][2]
    for span_start, span_end, page_number in page_spans:
        if start < span_end:
            start_page = page_number
            break

    end_page = page_spans[-1][2]
    for span_start, span_end, page_number in page_spans:
        end_page = page_number
        if end <= span_end:
            break
    return start_page, end_page


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, separators: list) -> list:
    """Splits a single block of text into overlapping chunks, preserving
    URLs and table-like line blocks."""
    text = _normalize_blank_lines(text)
    text, table_placeholder_map = _mask_tables(text)
    masked_text, url_placeholder_map = _mask_urls(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    masked_chunks = splitter.split_text(masked_text)
    chunks = [_unmask_urls(chunk, url_placeholder_map) for chunk in masked_chunks]
    return [_unmask_tables(chunk, table_placeholder_map) for chunk in chunks]


def chunk_documents(
    pages: list, chunk_size: int, chunk_overlap: int, separators: list
) -> list:
    """Chunks every document (all of its pages joined together) and
    returns a flat list of Chunk objects."""
    logger.info(
        "Chunking %d pages with chunk_size=%d, chunk_overlap=%d ...", len(pages), chunk_size, chunk_overlap
    )
    all_chunks = []
    for source_file, file_pages in itertools.groupby(pages, key=lambda page: page.source_file):
        file_pages = list(file_pages)
        logger.debug("Chunking %s (%d pages) ...", source_file, len(file_pages))

        full_text, page_spans = _join_pages(file_pages)
        headings = _detect_headings(full_text)
        heading_paths = _heading_paths(headings)
        marked_text = _insert_heading_markers(full_text, headings)
        chunk_separators = [HEADING_MARKER] + list(separators)

        raw_chunks = chunk_text(marked_text, chunk_size, chunk_overlap, chunk_separators)

        search_from = 0
        chunk_index = 0
        for raw_chunk in raw_chunks:
            cleaned_text = raw_chunk.replace(HEADING_MARKER, "").strip()
            if not cleaned_text:
                continue

            start = full_text.find(cleaned_text, search_from)
            if start == -1:
                start = full_text.find(cleaned_text)
            if start == -1:
                start = search_from
            end = start + len(cleaned_text)
            search_from = start

            if _looks_like_link_index(cleaned_text):
                continue

            page_number, page_end = _pages_for_span(page_spans, start, end)
            section_title = _section_title_for_offset(heading_paths, start)
            all_chunks.append(
                Chunk(
                    text=cleaned_text,
                    source_file=source_file,
                    page_number=page_number,
                    page_end=page_end,
                    chunk_index=chunk_index,
                    section_title=section_title,
                )
            )
            chunk_index += 1
    return all_chunks
