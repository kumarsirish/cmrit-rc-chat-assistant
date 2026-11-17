"""Admin tab: upload documents, mark one as superseding another, and
trigger a full index rebuild — the Streamlit-facing counterpart to the
backend primitives in document_loader.py/docs_metadata.py/
document_registry.py, which existed before this module but were never
wired to a UI.

Gated by a shared password (config.admin.password) remembered for the
rest of the browser session once entered correctly — this is a basic
deterrent, not a real auth system, matching the rest of the app's
no-real-login posture (see dashboard.DEFAULT_USER_EMAIL / rag._08_auth).
"""

from pathlib import Path

import streamlit as st

from rag._01_ingestion.document_loader import sanitize_topic_folder_name, save_uploaded_document
from rag._03_routing import docs_metadata, document_registry
from helper_scripts.ingest import run_ingestion

_SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "xlsx"]
_NO_SUPERSEDE_OPTION = "— none —"


def _existing_categories(source_directory: str) -> list:
    """Subfolder names directly under source_directory, each one an
    existing category an admin has already uploaded into. A brand-new
    deployment with nothing uploaded yet simply has none."""
    directory = Path(source_directory)
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


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
    password = st.text_input("Admin password", type="password")
    if st.button("Sign in"):
        if password == config.admin.password:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _render_upload_section(config):
    st.subheader("Upload a document")
    source_directory = config.documents.source_directory
    categories = _existing_categories(source_directory)

    category_options = ["— new category —"] + categories
    selected_option = st.selectbox("Category", category_options)
    selected_category = "" if selected_option == "— new category —" else selected_option
    typed_category = st.text_input("Or type a new category name")
    category = _resolve_category(selected_category, typed_category)

    uploaded_file = st.file_uploader("Document", type=_SUPPORTED_EXTENSIONS)

    supersedes = None
    if category:
        registry = document_registry.load_registry(config.documents.registry_file)
        active_in_category = [
            record.filename
            for record in registry
            if record.category == category and record.status == "active"
        ]
        if active_in_category:
            supersede_choice = st.selectbox(
                "This document replaces an existing document (optional)",
                [_NO_SUPERSEDE_OPTION] + active_in_category,
            )
            supersedes = None if supersede_choice == _NO_SUPERSEDE_OPTION else supersede_choice

    if st.button("Upload", disabled=not (uploaded_file and category)):
        filename = Path(uploaded_file.name).name
        save_uploaded_document(source_directory, category, filename, uploaded_file.getvalue())
        docs_metadata.add_document_to_topic(config.documents.metadata_file, category, filename)
        document_registry.register_upload(
            config.documents.registry_file,
            source_directory,
            filename,
            category,
            archived_subfolder_name=config.documents.archived_subfolder_name,
            supersedes=supersedes,
        )
        if supersedes:
            docs_metadata.remove_document_from_topic(config.documents.metadata_file, category, supersedes)
        st.success(
            f"Uploaded {filename} to category '{category}'"
            + (f", replacing {supersedes}" if supersedes else "")
            + ". Click 'Rebuild index' below to make it answerable."
        )


def _render_rebuild_section(config):
    st.subheader("Rebuild index")
    st.caption(
        "Re-scans every document in the source directory (skipping anything "
        "archived) and rebuilds the vector index from scratch. This reloads "
        "the embedding model and re-embeds every document, so it can take a "
        "while for a large document set."
    )
    if st.button("Rebuild index now"):
        with st.spinner("Rebuilding index — this may take a while..."):
            run_ingestion(config)
        st.success("Index rebuilt.")


def _render_documents_section(config):
    st.subheader("Documents")
    registry = document_registry.load_registry(config.documents.registry_file)
    if not registry:
        st.caption("No documents uploaded through this Admin tab yet.")
        return

    by_category = document_registry.list_documents_by_category(registry)
    for category in sorted(by_category):
        st.markdown(f"**{category}**")
        for record in by_category[category]:
            if record.status == "active":
                st.markdown(f"- {record.filename} — uploaded {record.upload_date}")
            else:
                st.markdown(
                    f"- ~~{record.filename}~~ — archived, superseded by "
                    f"{record.superseded_by} ({record.upload_date})"
                )


def render_admin_page(config) -> None:
    st.title("Admin")

    if not _render_password_gate(config):
        return

    _render_upload_section(config)
    st.divider()
    _render_rebuild_section(config)
    st.divider()
    _render_documents_section(config)
