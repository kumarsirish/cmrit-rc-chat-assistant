"""Entry point for the Streamlit dashboard.

Run via `streamlit run main.py` (or `make run` / `make run-debug`,
which just wrap that command) — never `python main.py` directly;
Streamlit's own process is what sets up the script-rerun/session
machinery the dashboard depends on.

This file is intentionally thin. The real orchestration lives
elsewhere:
  - rag/_10_ui/dashboard.py's main() is the actual app entry point
    (sidebar/nav setup, then the Chat/Admin pages).
  - rag/main_pipeline.py builds the shared RetrievalService/
    RagPipeline the app, helper_scripts/, and evaluation all run
    against — start there to see how routing, retrieval, and
    generation fit together.
  - helper_scripts/ingest.py builds the vector index (`make ingest`);
    it's a standalone batch step, not something the running app calls
    live except from the Admin page's "Rebuild index" button.
"""

import streamlit.runtime

from rag._09_common.logging_config import configure_logging

# Called before any other project import below, so whichever library
# first pulls in transformers/sentence-transformers (both known to
# turn on noisy INFO-level console logging as an import side effect)
# finds this process's logging already configured instead of
# configuring it themselves — see logging_config.py.
configure_logging()

if __name__ == "__main__":
    if not streamlit.runtime.exists():
        raise SystemExit("Run this with `streamlit run main.py`, not `python main.py`.")

    from rag._10_ui.dashboard import main as run_dashboard

    run_dashboard()
