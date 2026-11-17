"""Downloads the public Research Committee documents listed at
https://sites.google.com/cmrit.ac.in/research-committee/downloads into
knowledge_documents, so they can be ingested like any other document
(see rag/_01_ingestion/document_loader.py). Every linked document is
registered the same way the Admin UI's upload flow registers one (see
rag/_03_routing/docs_metadata.py, rag/_03_routing/document_registry.py),
so each downloaded form shows up in the Admin tab's document list and
is individually topic-matchable at query time.

Run it, then `make ingest` (or the Admin tab's "Rebuild index" button)
to actually index what it downloaded:

    python helper_scripts/download_rc_documents.py

Every link on the page is either a native Google Docs/Sheets/Slides
document or a generic Drive-hosted file, and the page itself is public
(shared "anyone with the link") — each is exported/downloaded in a
native, ingestible format (.docx/.xlsx/.pptx for Google's own editor
formats, whatever Drive serves for a generic uploaded file, usually
.pdf here) rather than scraped as rendered HTML, so the document's
actual content survives intact.

Safe to re-run: a later run just re-downloads and overwrites each
file in place and refreshes its document_registry upload_date (see
DocumentRegistry.register_upload's overwrite handling) rather than
creating duplicates, so running this again after the page adds a new
form, or a Google Doc on it is edited, picks up the change without
hand-curating what's already there.
"""

import logging
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup

from rag._01_ingestion.document_loader import sanitize_topic_folder_name
from rag._03_routing import docs_metadata, document_registry
from rag._09_common.config import AppConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SOURCE_PAGE = "https://sites.google.com/cmrit.ac.in/research-committee/downloads"
# Folder (under documents.source_directory) every downloaded file is
# saved into. Each file is still registered under its own individual
# docs-metadata.json topic (see _download_one) — this is just the
# physical folder, not a grouping of what's topic-matchable together.
DEFAULT_CATEGORY = "Research Committee"

_REQUEST_TIMEOUT_SECONDS = 60
# A plain "python-requests" UA gets a different (sometimes stripped-
# down) response from Google Sites than a real browser would — this
# matches what a browser sends.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Every document-shaped format this script knows how to resolve a
# Google Docs/Drive link into. Kept in sync with
# document_loader._LOADERS_BY_EXTENSION by hand (same small, stable
# set admin_page.py's uploader also hand-maintains) rather than
# importing that private module constant across files.
_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx", ".pptx"}

_DOCUMENT_ID_PATTERN = re.compile(r"/document/d/([^/]+)")
_SPREADSHEET_ID_PATTERN = re.compile(r"/spreadsheets/d/([^/]+)")
_PRESENTATION_ID_PATTERN = re.compile(r"/presentation/d/([^/]+)")
_DRIVE_FILE_ID_PATTERN = re.compile(r"/file/d/([^/]+)")

_UNSAFE_FILENAME_CHARS_PATTERN = re.compile(r"[^\w\s\-().,&]")
_CONTENT_DISPOSITION_FILENAME_PATTERN = re.compile(r'filename="?([^";]+)"?')


class DownloadLink:
    """One resolved document link: label is the human-readable text
    from the downloads page (used to name the saved file and as its
    docs-metadata.json topic); source_url is the original page the
    link pointed at (used as that topic's canonical link); export_url
    is where to actually fetch the file's bytes from; extension is
    known upfront for a native Google Docs/Sheets/Slides export, or
    None for a generic Drive file (discovered from the response's
    Content-Disposition header instead — see _download_one)."""

    def __init__(self, label: str, source_url: str, export_url: str, extension: str = None):
        self.label = label
        self.source_url = source_url
        self.export_url = export_url
        self.extension = extension


def _drive_file_id_from_open_url(url: str) -> str:
    """Extracts the id= query parameter from a legacy
    drive.google.com/open?id=... URL."""
    query = parse_qs(urlparse(url).query)
    values = query.get("id")
    return values[0] if values else None


def classify_link(href: str, label: str) -> DownloadLink:
    """Returns a DownloadLink for href, or None if href isn't a
    recognized Google Docs/Drive document link (e.g. an internal Sites
    navigation link, a mailto:, an unrelated external site)."""
    if match := _DOCUMENT_ID_PATTERN.search(href):
        file_id = match.group(1)
        return DownloadLink(
            label, href, f"https://docs.google.com/document/d/{file_id}/export?format=docx", ".docx"
        )
    if match := _SPREADSHEET_ID_PATTERN.search(href):
        file_id = match.group(1)
        return DownloadLink(
            label, href, f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx", ".xlsx"
        )
    if match := _PRESENTATION_ID_PATTERN.search(href):
        file_id = match.group(1)
        return DownloadLink(
            label, href, f"https://docs.google.com/presentation/d/{file_id}/export/pptx", ".pptx"
        )
    if match := _DRIVE_FILE_ID_PATTERN.search(href):
        file_id = match.group(1)
        return DownloadLink(label, href, f"https://drive.google.com/uc?export=download&id={file_id}")
    if "drive.google.com/open" in href:
        file_id = _drive_file_id_from_open_url(href)
        if file_id:
            return DownloadLink(label, href, f"https://drive.google.com/uc?export=download&id={file_id}")
    return None


def find_download_links(page_url: str, session: requests.Session) -> list:
    """Scrapes page_url for every recognized Google Docs/Drive link,
    returning one DownloadLink per distinct export_url (a document
    linked more than once on the page, e.g. in a page header and
    again in a list, is only returned once)."""
    response = session.get(page_url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    seen_export_urls = set()
    for anchor in soup.find_all("a", href=True):
        label = anchor.get_text(strip=True)
        if not label:
            continue
        link = classify_link(anchor["href"], label)
        if link is None or link.export_url in seen_export_urls:
            continue
        seen_export_urls.add(link.export_url)
        links.append(link)
    return links


def _sanitize_filename_stem(label: str) -> str:
    name = _UNSAFE_FILENAME_CHARS_PATTERN.sub("", label.strip()).strip()
    return name or "document"


def _extension_from_response(response: requests.Response) -> str:
    """Recovers the real file extension for a generic Drive file
    (DownloadLink.extension is None for these) from the
    Content-Disposition header Drive sends alongside the actual
    bytes."""
    content_disposition = response.headers.get("Content-Disposition", "")
    match = _CONTENT_DISPOSITION_FILENAME_PATTERN.search(content_disposition)
    if not match:
        return None
    return Path(match.group(1)).suffix.lower()


def _download_one(config: AppConfig, session: requests.Session, link: DownloadLink, category: str) -> bool:
    """Downloads one link into documents.source_directory/category,
    registers it under its own docs-metadata.json topic (link.label)
    and in the document registry, and returns True on success. Logs
    and returns False on any failure (network error, a Drive
    "can't scan this file for viruses" HTML interstitial instead of
    real content, or an unrecognized/unsupported resulting extension)
    — one bad link shouldn't abort downloading the rest, same pattern
    as document_loader.load_all_documents skipping one bad file.
    """
    try:
        response = session.get(
            link.export_url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to download '%s': %s", link.label, exc)
        return False

    extension = link.extension or _extension_from_response(response)
    content_type = response.headers.get("Content-Type", "")
    if extension not in _SUPPORTED_EXTENSIONS or content_type.startswith("text/html"):
        logger.warning(
            "Skipping '%s': got an unsupported or unrecognized response "
            "(content-type=%s, extension=%s) — Drive sometimes serves an "
            "HTML interstitial instead of the file itself for a link that "
            "needs a manual confirmation click; download it by hand if so.",
            link.label,
            content_type,
            extension,
        )
        return False

    filename = f"{_sanitize_filename_stem(link.label)}{extension}"
    category_directory = Path(config.documents.source_directory) / sanitize_topic_folder_name(category)
    category_directory.mkdir(parents=True, exist_ok=True)
    (category_directory / filename).write_bytes(response.content)

    docs_metadata.add_document_to_topic(
        config.documents.metadata_file, topic=link.label, filename=filename, link=link.source_url
    )
    document_registry.register_upload(
        config.documents.registry_file,
        config.documents.source_directory,
        filename,
        category,
        archived_subfolder_name=config.documents.archived_subfolder_name,
    )
    logger.info("Downloaded '%s' -> %s", link.label, category_directory / filename)
    return True


def download_rc_documents(
    config: AppConfig = None, page_url: str = DEFAULT_SOURCE_PAGE, category: str = DEFAULT_CATEGORY
) -> None:
    if config is None:
        config = load_config()

    session = requests.Session()
    logger.info("Scraping download links from %s ...", page_url)
    links = find_download_links(page_url, session)
    logger.info("Found %d document link(s).", len(links))

    succeeded = 0
    for link in links:
        if _download_one(config, session, link, category):
            succeeded += 1

    logger.info(
        "Done. Downloaded %d/%d document(s) into %s/%s. Run `make ingest` "
        "(or the Admin tab's 'Rebuild index' button) to index them.",
        succeeded,
        len(links),
        config.documents.source_directory,
        sanitize_topic_folder_name(category),
    )


if __name__ == "__main__":
    download_rc_documents()
