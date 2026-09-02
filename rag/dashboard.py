"""Streamlit dashboard for the CMRIT college policy assistant.

This is the only module that contains UI/dashboard code; every other
module is a plain Python component that this module calls into.
`app.py` is just the thin entry point Streamlit runs.
"""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag.config import load_config
from rag.embeddings import EmbeddingGenerator
from rag.llm_client import LlmClient
from rag.rag_pipeline import RagPipeline
from rag.retriever import Retriever
from rag.session_store import SessionStore
from rag.user_identity import is_valid_email, normalize_email
from rag.vector_store import VectorStore

st.set_page_config(page_title="CMRIT Policy Assistant", page_icon="📘", layout="wide")

# Streamlit's default centered layout leaves large blank margins on
# either side of the chat. Constrain the content to a comfortable
# reading width instead of the full window, and trim the default top
# padding above the title.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Enabled by `make run-debug` (sets RAG_DEBUG=true). Shows the raw
# query, every retrieved chunk's source document/page, and its
# similarity score, so retrieval behaviour can be inspected live.
DEBUG_ENABLED = os.environ.get("RAG_DEBUG", "false").lower() == "true"


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_rag_pipeline():
    config = get_config()
    embedding_generator = EmbeddingGenerator(config.embedding.model_name)
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=config.retrieval.top_k,
        min_relevance_score=config.retrieval.min_relevance_score,
    )
    llm_client = LlmClient(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        temperature=config.ollama.temperature,
        timeout_seconds=config.ollama.request_timeout_seconds,
        num_thread=config.ollama.num_thread,
    )
    return RagPipeline(retriever=retriever, llm_client=llm_client)


@st.cache_resource
def get_session_store():
    config = get_config()
    return SessionStore(config.database, config.session.timeout_minutes)


def start_new_session(session_store, user_email):
    """Starts a fresh backend session and wipes this browser tab's chat
    context — used both for a brand-new visitor and for resetting an
    expired/inactive session.
    """
    session_id = session_store.create_session(user_email)
    st.session_state.session_id = session_id
    st.session_state.chat_history = []
    st.session_state.last_active_at = datetime.now()


def touch_session(session_store):
    session_store.touch_session(st.session_state.session_id)
    st.session_state.last_active_at = datetime.now()


def ensure_active_session(session_store, timeout_minutes):
    """Requires an email to already be in st.session_state.user_email.

    This is the authoritative (database-backed) expiry check, run on
    every real interaction. The live countdown fragment below also
    resets an idle session proactively, without waiting for the next
    click, but this check is what guarantees an expired session is
    never used even if that fragment hasn't fired yet.
    """
    user_email = st.session_state.user_email

    if "session_id" not in st.session_state:
        start_new_session(session_store, user_email)
        return

    if session_store.is_expired(st.session_state.session_id):
        st.info(
            f"Your session timed out after {timeout_minutes} minutes of "
            "inactivity. Starting a new session."
        )
        start_new_session(session_store, user_email)


@st.fragment(run_every=1)
def render_session_countdown(session_store, config):
    """Live, self-updating idle countdown.

    Runs on its own 1-second timer (via st.fragment) independent of
    user interaction. Once idle time crosses the warning threshold it
    shows a countdown; once idle time reaches the full timeout it
    resets the session (wiping chat context) and reruns the app,
    without needing the user to do anything first.
    """
    timeout_seconds = config.session.timeout_minutes * 60
    warning_seconds = config.session.warning_seconds_before_timeout
    idle_seconds = (datetime.now() - st.session_state.last_active_at).total_seconds()
    remaining_seconds = timeout_seconds - idle_seconds

    if remaining_seconds <= 0:
        start_new_session(session_store, st.session_state.user_email)
        st.rerun()
        return

    if remaining_seconds <= warning_seconds:
        st.warning(
            f"⏳ Idle for a while — this session will reset in "
            f"{int(remaining_seconds) + 1} seconds and your chat "
            "history will be cleared."
        )


def render_email_gate():
    """Asks for the user's email before any policy question can be asked."""
    st.title("CMRIT College Policy Assistant")
    st.write(
        "Please enter your college email address to start a session. "
        "Your conversation history is tied to this address."
    )
    with st.form("email_gate_form"):
        email_input = st.text_input("Email address")
        submitted = st.form_submit_button("Continue")

    if submitted:
        if is_valid_email(email_input):
            st.session_state.user_email = normalize_email(email_input)
            st.rerun()
        else:
            st.error("Please enter a valid email address, e.g. name@cmrit.ac.in")


def render_sidebar(config):
    with st.sidebar:
        st.header("About")
        st.write(
            "This assistant answers questions about CMRIT college policies "
            "using only the official policy documents. It will not answer "
            "questions unrelated to those policies."
        )
        st.divider()
        st.caption(f"Signed in as: {st.session_state.user_email}")
        st.caption(f"Model: {config.ollama.model}")
        st.caption(f"Session timeout: {config.session.timeout_minutes} minutes")
        if DEBUG_ENABLED:
            st.caption("Debug mode: ON")
        if st.button("Start a new session"):
            session_store = get_session_store()
            start_new_session(session_store, st.session_state.user_email)
            st.rerun()
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()


TOP_SOURCES_SHOWN = 2


def render_sources(sources, config, key_suffix):
    """Shows links to the top TOP_SOURCES_SHOWN most relevant source
    PDFs below the answer (the answer text itself carries no citations
    — see _strip_inline_citations in rag_pipeline.py).
    """
    if not sources:
        return
    st.caption("Sources:")
    for rank, source in enumerate(sources[:TOP_SOURCES_SHOWN], start=1):
        pdf_path = Path(config.documents.source_directory) / source.source_file

        if not pdf_path.exists():
            st.caption(f"{rank}. {source.source_file}, page {source.page_number}")
            continue

        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(
            label=f"📄 {source.source_file} (page {source.page_number})",
            data=pdf_bytes,
            file_name=source.source_file,
            mime="application/pdf",
            key=f"source-{key_suffix}-{rank}",
        )


def render_debug_panel(question, sources):
    """Debug-only: shows the query and every retrieved chunk with its
    source document, page, and similarity score. Only rendered when
    RAG_DEBUG=true (set by `make run-debug`).
    """
    if not DEBUG_ENABLED:
        return
    with st.expander("🐛 Debug: retrieval details"):
        st.write(f"**Query:** {question}")
        if not sources:
            st.write("No chunks passed the relevance threshold.")
            return
        for rank, source in enumerate(sources, start=1):
            st.write(
                f"{rank}. **{source.source_file}**, page {source.page_number} "
                f"— similarity: {source.relevance_score:.3f} "
                f"— matched via: {source.match_type}"
            )
            st.caption(source.text_preview)


def _build_conversation_history(chat_history, max_messages):
    """Turns the session's chat_history so far into the {role, content}
    list the LLM expects, capped to the most recent max_messages turns
    so a long session doesn't keep growing the prompt indefinitely.
    """
    recent_messages = chat_history[-max_messages:]
    return [
        {"role": message["role"], "content": message["content"]}
        for message in recent_messages
    ]


def main():
    config = get_config()
    session_store = get_session_store()

    if "user_email" not in st.session_state:
        render_email_gate()
        return

    ensure_active_session(session_store, config.session.timeout_minutes)
    render_sidebar(config)

    st.title("CMRIT College Policy Assistant")
    st.caption(
        "Ask a question about CMRIT college policies. Answers are grounded "
        "in the official policy documents."
    )
    render_session_countdown(session_store, config)

    for message_index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []), config, f"history-{message_index}")
                render_debug_panel(message.get("question", ""), message.get("sources", []))

    question = st.chat_input("Ask about a college policy...")
    if question:
        # Captured before this turn is appended, so it holds only
        # earlier turns — the current question is sent separately.
        conversation_history = _build_conversation_history(
            st.session_state.chat_history, config.session.max_history_messages
        )

        touch_session(session_store)
        session_store.add_message(st.session_state.session_id, "user", question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Checking the policy documents..."):
                rag_pipeline = get_rag_pipeline()
                result = rag_pipeline.answer(question, conversation_history=conversation_history)
            st.write(result.answer_text)
            render_sources(result.sources, config, f"new-{len(st.session_state.chat_history)}")
            render_debug_panel(question, result.sources)

        session_store.add_message(
            st.session_state.session_id, "assistant", result.answer_text
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.answer_text,
                "sources": result.sources,
                "question": question,
            }
        )
