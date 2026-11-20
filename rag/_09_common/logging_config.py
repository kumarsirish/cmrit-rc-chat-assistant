"""Central place that configures logging for a running process — the
live Streamlit app, or any standalone script under helper_scripts/.

Without this, console verbosity ends up decided by whichever
third-party library happens to call logging.basicConfig() (or set its
own logger's level) first as an import side effect — transformers and
sentence-transformers are both known to do this, silently turning on
INFO-level "Batches: 100%|..." progress bars and noisy httpx request
logs the moment they're imported, regardless of anything this app
wants. basicConfig() only has any effect the FIRST time it's called
with no existing root handler, so calling this as early as possible in
a process's entry point (before importing anything that pulls those
libraries in) is what makes the suppression below actually stick.

This app's own "rag.*" loggers are left at the root's WARNING level
too, not forced to INFO — per-query retrieval detail now lives in the
structured JSON log (see query_logger.py) and, for ingestion, in the
Admin UI's own progress display, so there's nothing left that needs
this process's console for routine, successful operation. Warnings and
errors still print normally.
"""

import logging

_NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "filelock",
    "urllib3",
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
