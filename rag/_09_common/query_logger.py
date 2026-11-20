"""Per-query structured JSON debug log.

One line of JSON per question answered (JSON Lines — one self-
contained JSON object per line, not a single array), covering the
whole pipeline: the question, every candidate chunk dense/BM25 search
considered, the reranker's score for every chunk it scored, the
chunks that actually survived to the LLM, the LLM's raw answer, and
how long each stage took. Meant to be read later when a bad answer
needs to be traced back to which stage went wrong — this never
renders anywhere in the app itself (see dashboard.render_debug_panel
for that, which is unrelated and reads live chunks/sources directly).

JSON Lines rather than one JSON array: each line is valid on its own
and the file can be safely appended to while the app is running,
without ever needing to close/rewrite a top-level array.

A new file is started every time this module is imported — i.e. once
per app process/launch, since Python only runs a module's top-level
code once per process — named with that moment's timestamp, so
restarting the app never appends to or overwrites a previous run's log.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from rag._09_common.config import PROJECT_ROOT

_LOG_DIRECTORY = PROJECT_ROOT / "logs"
# dd-mm-yyyy.HH:MM — the colon is a valid filename character on this
# app's deployment target (Linux); if this is ever deployed where the
# log directory is read back through a Windows/NTFS path (e.g. Windows
# Explorer over a WSL \\wsl$ share), NTFS forbids ":" in filenames and
# this format would need adjusting.
_LOG_PATH = _LOG_DIRECTORY / f"query_log_{datetime.now():%d-%m-%Y.%H:%M}.jsonl"

logger = logging.getLogger(__name__)


def log_query(record: dict) -> None:
    """Appends record as one line of JSON to this process's query log
    file. Never raises — a logging failure must not break answering a
    question."""
    try:
        _LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not write to query log %s: %s", _LOG_PATH, exc)
