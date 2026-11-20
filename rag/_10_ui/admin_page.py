"""Admin tab: upload documents, mark one as superseding another,
trigger a full index rebuild, and clear chat history — the
Streamlit-facing counterpart to the backend primitives in
document_loader.py/document_metadata.py/session_store.py,
most of which existed before this module but were never wired to a UI.

Gated by a shared password (config.admin.password) remembered for the
rest of the browser session once entered correctly — this is a basic
deterrent, not a real auth system, matching the rest of the app's
no-real-login posture (see rag._08_auth).
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag._01_ingestion.document_loader import (
    extract_zip_archive,
    find_document_files,
    list_categories,
    sanitize_topic_folder_name,
    save_uploaded_document,
)
from rag._03_routing import document_metadata
from rag._08_auth.session_store import SessionStore
from rag._09_common.system_info import render_system_info_page
from rag._10_ui.page_header import render_page_title
from helper_scripts.ingest import load_ingestion_stats, run_ingestion

_SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "xlsx", "pptx"]
_UPLOAD_TYPES = _SUPPORTED_EXTENSIONS + ["zip"]
_NO_SUPERSEDE_OPTION = "— none —"

_TIMESTAMP_FORMAT = "%d-%m-%Y, %H:%M:%S"


def _format_timestamp(unix_timestamp: float) -> str:
    return datetime.fromtimestamp(unix_timestamp).strftime(_TIMESTAMP_FORMAT)


def _resolve_category(selected_category: str, typed_category: str) -> str:
    """The free-text "new category" field wins whenever non-empty, so
    typing a new name always creates that category rather than silently
    falling back to whatever the selectbox happened to have focused."""
    typed_category = (typed_category or "").strip()
    return typed_category if typed_category else selected_category


def _render_password_gate(config) -> bool:
    if st.session_state.get("admin_authenticated"):
        return True

    st.subheader("Admin sign-in")
    # Streamlit defaults password-type inputs to autocomplete="new-password",
    # which tells the browser this is a signup field and makes it offer to
    # generate/save a fresh strong password — misleading here, since this
    # is a login field for an existing shared password. "current-password"
    # is the correct MDN autocomplete token for that case.
    password = st.text_input("Admin password", type="password", autocomplete="current-password")
    if st.button("Sign in"):
        if password == config.admin.password:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _register_single_upload(config, source_directory, category, filename, supersedes):
    document_metadata.register_upload(
        config.documents.documents_file,
        source_directory,
        filename,
        category,
        archived_subfolder_name=config.documents.archived_subfolder_name,
        supersedes=supersedes,
    )


def _render_upload_section(config):
    st.caption(
        "Upload a single PDF/Word/text/Excel/PowerPoint file, or a .zip containing "
        "several — the zip is extracted into the chosen category and then "
        "discarded; any file inside it that matches an existing filename "
        "overwrites that document in place."
    )
    source_directory = config.documents.source_directory
    categories = list_categories(source_directory)

    category_options = ["— new category —"] + categories
    selected_option = st.selectbox("Category", category_options)
    selected_category = "" if selected_option == "— new category —" else selected_option
    typed_category = st.text_input("Or type a new category name")
    category = _resolve_category(selected_category, typed_category)

    uploaded_file = st.file_uploader("Document or .zip", type=_UPLOAD_TYPES)
    is_zip = bool(uploaded_file) and uploaded_file.name.lower().endswith(".zip")

    supersedes = None
    if category and not is_zip:
        records = document_metadata.load_documents(config.documents.documents_file)
        active_in_category = [
            record.filename
            for record in records
            if record.category == category and record.status == "active"
        ]
        if active_in_category:
            supersede_choice = st.selectbox(
                "This document replaces an existing document (optional)",
                [_NO_SUPERSEDE_OPTION] + active_in_category,
            )
            supersedes = None if supersede_choice == _NO_SUPERSEDE_OPTION else supersede_choice

    if st.button("Upload", disabled=not (uploaded_file and category)):
        if is_zip:
            extracted_filenames = extract_zip_archive(source_directory, category, uploaded_file.getvalue())
            for extracted_filename in extracted_filenames:
                _register_single_upload(config, source_directory, category, extracted_filename, supersedes=None)
            if extracted_filenames:
                st.success(
                    f"Extracted {len(extracted_filenames)} document(s) from "
                    f"{uploaded_file.name} into category '{category}': "
                    f"{', '.join(extracted_filenames)}. Click 'Rebuild index' "
                    "below to make them answerable."
                )
            else:
                st.warning(f"No supported documents found inside {uploaded_file.name}.")
        else:
            filename = Path(uploaded_file.name).name
            save_uploaded_document(source_directory, category, filename, uploaded_file.getvalue())
            _register_single_upload(config, source_directory, category, filename, supersedes)
            st.success(
                f"Uploaded {filename} to category '{category}'"
                + (f", replacing {supersedes}" if supersedes else "")
                + ". Click 'Rebuild index' below to make it answerable."
            )


def _rebuild_progress_path(config) -> Path:
    """Where the running rebuild's progress is persisted — a sibling of
    documents.ingestion_stats_file. Deliberately a file, not
    st.session_state: switching between the Admin/Chat/System Info
    st.navigation pages and back was observed to lose session_state
    set here (even plain, non-widget-bound entries), wiping the
    in-progress display entirely. A file on disk is immune to that by
    construction — whichever page re-renders this section just reads
    whatever's currently there, same as config itself is read from
    disk rather than carried in memory across the page switch.
    """
    return Path(config.documents.ingestion_stats_file).with_name("rebuild_progress.json")


def _write_rebuild_progress(path: Path, state: dict) -> None:
    # Write-to-temp-then-rename so a concurrent read (the polling
    # fragment, possibly mid-tick) never sees a half-written file.
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as progress_file:
        json.dump(state, progress_file)
    tmp_path.replace(path)


def _read_rebuild_progress(path: Path) -> dict:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as progress_file:
            return json.load(progress_file)
    except (OSError, json.JSONDecodeError):
        return None


def _start_rebuild(config, estimated_total_seconds: float) -> None:
    """Kicks off run_ingestion() on a background thread that reports
    its own progress by writing _rebuild_progress_path(config) after
    every stage/batch (see helper_scripts.ingest.run_ingestion's
    progress_callback) — the ONLY thing _render_rebuild_progress reads
    from, independent of Streamlit session/page state (see
    _rebuild_progress_path's docstring for why).

    The dict being written is a plain local variable captured by the
    worker thread's own closure and only ever mutated from within that
    one thread — the UI side never touches it directly, only the file
    it's serialized to, so there's no cross-thread mutable-state hazard
    beyond the file I/O itself (write-temp-then-rename makes that safe
    too).
    """
    progress_path = _rebuild_progress_path(config)
    start = time.perf_counter()
    state = {
        "status": "running",  # "running" | "done" | "error"
        "message": "Starting...",
        "fraction": 0.0,
        "log_lines": [],
        "start_time": start,
        "estimated_total_seconds": estimated_total_seconds,
        "error": None,
    }
    _write_rebuild_progress(progress_path, state)

    def on_progress(message, fraction):
        state["message"] = message
        state["fraction"] = fraction
        state["log_lines"].append(f"[{time.perf_counter() - start:.0f}s] {message}")
        _write_rebuild_progress(progress_path, state)

    def worker():
        try:
            run_ingestion(config, progress_callback=on_progress)
            # Drops every @st.cache_resource-cached object (the
            # assembled RetrievalService/topic matchers, the category
            # list, ...) so the next chat interaction — in any session,
            # since this cache is process-wide — rebuilds them from
            # what's on disk now. Without this, a category/document
            # added, renamed, or removed by this rebuild wouldn't take
            # effect until the process restarts.
            st.cache_resource.clear()
            state["status"] = "done"
            state["fraction"] = 1.0
        except Exception as exc:
            state["status"] = "error"
            state["error"] = str(exc)
        _write_rebuild_progress(progress_path, state)

    threading.Thread(target=worker, daemon=True).start()


@st.fragment(run_every=1)
def _render_rebuild_progress(config):
    """Re-reads the progress file every second (independent of any
    other interaction with the page — see st.fragment's run_every) and
    renders the bar/elapsed time/log from whatever's currently there,
    so the display keeps advancing continuously even during a long
    quiet stretch between the worker thread's own progress_callback
    calls — and keeps working correctly even if the user switches to
    another page and back mid-rebuild, since nothing here depends on
    this specific script run having been the one that started it."""
    progress_path = _rebuild_progress_path(config)
    state = _read_rebuild_progress(progress_path)
    if state is None:
        return

    elapsed_seconds = time.perf_counter() - state["start_time"]
    if state["status"] == "done":
        display_fraction = 1.0
    else:
        # Ticks every second regardless of whether the worker thread
        # has written anything new, so the bar keeps visibly moving
        # through a long quiet stretch instead of looking frozen —
        # real per-stage fraction (state["fraction"]) still wins once
        # it's ahead, e.g. once the run genuinely finishes faster than
        # the estimate.
        time_based_fraction = min(elapsed_seconds / state["estimated_total_seconds"], 0.99)
        display_fraction = max(state["fraction"], time_based_fraction)

    st.progress(min(max(display_fraction, 0.0), 1.0), text=state["message"])
    st.caption(f"Elapsed: {elapsed_seconds:.0f}s (estimated total: ~{state['estimated_total_seconds']:.0f}s)")
    st.code("\n".join(state["log_lines"]), language=None)

    if state["status"] == "error":
        st.error(f"Rebuild failed: {state['error']}")
        if st.button("Dismiss", key="dismiss_rebuild_error"):
            progress_path.unlink(missing_ok=True)
            st.rerun()
    elif state["status"] == "done":
        st.success(f"Index rebuilt in {elapsed_seconds:.0f}s.")
        if st.button("Dismiss", key="dismiss_rebuild_done"):
            progress_path.unlink(missing_ok=True)
            st.rerun()


def _render_rebuild_section(config):
    st.subheader("Rebuild index")
    st.caption(
        "Re-scans every document in the source directory (skipping anything "
        "archived) and rebuilds the vector index from scratch. This reloads "
        "the embedding model and re-embeds every document, so it can take a "
        "while for a large document set."
    )

    stats = load_ingestion_stats(config)
    if stats.get("last_duration_seconds"):
        st.caption(
            f"Estimated time: ~{stats['last_duration_seconds']:.0f}s "
            f"(based on the last rebuild, {stats.get('last_run_at', 'unknown date')})"
        )

    if _read_rebuild_progress(_rebuild_progress_path(config)) is None:
        # Fallback estimate (~4 minutes) for a brand-new deployment
        # with no prior successful rebuild to measure from yet.
        estimated_total_seconds = stats.get("last_duration_seconds") or 240.0
        if st.button("Rebuild index now"):
            _start_rebuild(config, estimated_total_seconds)
            st.rerun()
    else:
        _render_rebuild_progress(config)


def _render_last_vectorization_status(config):
    sqlite_path = Path(config.chroma.persist_directory) / "chroma.sqlite3"
    if not sqlite_path.exists():
        st.markdown("**Documents have not been vectorized yet** — use 'Rebuild index now' below.")
        return
    st.markdown(f"**Last vectorized:** {_format_timestamp(sqlite_path.stat().st_mtime)}")


def _render_documents_section(config):
    _render_last_vectorization_status(config)

    source_directory = config.documents.source_directory
    try:
        document_paths = find_document_files(source_directory, config.documents.archived_subfolder_name)
    except FileNotFoundError:
        document_paths = []

    records = document_metadata.load_documents(config.documents.documents_file)

    if not document_paths:
        st.caption("No documents found.")
    else:
        # filename -> canonical link (same source as the per-source
        # "link:" caption shown under a chat answer's citations) — a
        # document with no matching record, or whose record has no
        # link (e.g. one uploaded via this Admin tab without a link
        # collected), just shows plain text.
        filename_topic_map = document_metadata.active_filename_topic_map(records)

        source_root = Path(source_directory)
        by_category = {}
        for path in document_paths:
            category = (
                path.parent.relative_to(source_root).parts[0] if path.parent != source_root else "(uncategorized)"
            )
            by_category.setdefault(category, []).append(path)

        for category in sorted(by_category):
            st.markdown(f"**{category}**")
            for path in sorted(by_category[category]):
                timestamp = _format_timestamp(path.stat().st_mtime)
                topic = filename_topic_map.get(path.name.strip().casefold())
                if topic and topic.link:
                    st.markdown(f"- [{path.name}]({topic.link}) — {timestamp}")
                else:
                    st.markdown(f"- {path.name} — {timestamp}")

    archived_records = [record for record in records if record.status == "archived"]
    if archived_records:
        st.markdown("**Archived (superseded)**")
        for record in archived_records:
            st.markdown(
                f"- ~~{record.filename}~~ — superseded by "
                f"{record.superseded_by} ({record.upload_date})"
            )


def _render_clear_history_section(config):
    st.subheader("Clear chat history")
    st.caption(
        "Permanently deletes every user's session and chat history — "
        "including what shows up on the Recent Queries page. This cannot "
        "be undone."
    )
    confirmed = st.checkbox("I understand this will permanently delete all chat history for every user.")
    if st.button("Delete all history", disabled=not confirmed):
        session_store = SessionStore(config.database, config.session.timeout_minutes)
        session_store.clear_all_history()
        st.success("All chat history has been deleted.")


def _render_recent_queries_tab(config) -> None:
    """Shows the most recent answered questions across EVERY user's
    session (see SessionStore.get_recent_queries) — unlike the Chat
    page, which only ever shows the viewer's own session history, this
    is deliberately shared/global, same database rows everyone else
    sees here too. Moved here from its own top-level page so it's one
    less entry in the sidebar nav."""
    st.caption("The most recently answered questions, across all users.")

    session_store = SessionStore(config.database, config.session.timeout_minutes)
    recent_queries = session_store.get_recent_queries()
    if not recent_queries:
        st.caption("No queries have been answered yet.")
        return

    for recent_query in recent_queries:
        st.markdown(f"**Q:** {recent_query.question}")
        st.write(recent_query.answer)
        timestamp_column, rating_column = st.columns([3, 1])
        with timestamp_column:
            st.caption(recent_query.answered_at.strftime("%Y-%m-%d %H:%M:%S"))
        with rating_column:
            st.caption(_format_rating(recent_query.rating))
        st.divider()


def _format_rating(rating) -> str:
    """"Not rated" when no user has rated this answer yet, otherwise a
    star/empty-star readout, e.g. "★★★☆☆ (3/5)"."""
    if rating is None:
        return "Not rated"
    return f"{'★' * rating}{'☆' * (5 - rating)} ({rating}/5)"


def render_admin_page(config, user_email: str) -> None:
    render_page_title("Admin")

    if not _render_password_gate(config):
        return

    (
        upload_tab,
        existing_documents_tab,
        misc_tab,
        system_info_tab,
        recent_queries_tab,
    ) = st.tabs(
        ["Upload Documents", "Existing Documents", "Misc Actions", "System Info", "Recent Queries"]
    )

    with upload_tab:
        _render_upload_section(config)

    with existing_documents_tab:
        _render_documents_section(config)
        st.divider()
        _render_rebuild_section(config)

    with misc_tab:
        _render_clear_history_section(config)

    with system_info_tab:
        render_system_info_page(config, user_email)

    with recent_queries_tab:
        _render_recent_queries_tab(config)
