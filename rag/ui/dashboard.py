"""Streamlit dashboard for the CMRIT college policy assistant.

This is the only module that contains UI/dashboard code; every other
module is a plain Python component that this module calls into.
`app.py` is just the thin entry point Streamlit runs.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag.auth.session_store import SessionStore
from rag.auth.user_identity import normalize_email
from rag.common.config import PROJECT_ROOT, load_config
from rag.common.system_info import render_system_info_page
from rag.generation.llm_client import LlmClient
from rag.generation.rag_pipeline import RagPipeline
from rag.indexing.bm25_index import BM25Index
from rag.indexing.docs_metadata import load_doc_topics
from rag.indexing.vector_store import VectorStore
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import TRACE_LOGGER_NAME, Retriever

APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.png"

st.set_page_config(page_title="chatbot-policies", page_icon=str(APP_ICON_PATH), layout="wide")

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

RETRIEVAL_DEBUG_LOG_PATH = PROJECT_ROOT / "logs" / "retrieval_debug.log"

# The email gate is skipped for now; every session is attributed to
# this address instead.
DEFAULT_USER_EMAIL = normalize_email("sirish.k@cmrit.ac.in")


def render_page_title(title):
    """Shows the app icon beside a page title, instead of plain st.title."""
    icon_column, title_column = st.columns([1, 12], vertical_alignment="center")
    with icon_column:
        st.image(str(APP_ICON_PATH))
    with title_column:
        st.title(title)


@st.cache_resource
def _configure_retrieval_debug_logging():
    """Attaches a file handler to Retriever's trace logger (see
    retriever.py's _log_debug_trace) so every query's retrieval trace —
    detected topic, mapped document, retrieval mode
    (MAPPED_DOCUMENT/GLOBAL), candidates, final chunks and their source
    metadata — is written to logs/retrieval_debug.log on every run,
    debug or not. RAG_DEBUG=true additionally surfaces the same trace
    in the in-app debug panel (see render_debug_panel).

    @st.cache_resource makes this run its body only once per process:
    Streamlit re-executes this module on every rerun (i.e. after every
    question), and without caching that would attach a duplicate
    handler — and duplicate log lines — each time.
    """
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    RETRIEVAL_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(RETRIEVAL_DEBUG_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("\n[%(asctime)s]\n%(message)s"))
    trace_logger.addHandler(handler)
    trace_logger.setLevel(logging.DEBUG)
    # Keep this out of the console/root logger — it's a lot of detail
    # per query, meant for the dedicated log file only.
    trace_logger.propagate = False
    return trace_logger


_configure_retrieval_debug_logging()


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_vector_store():
    config = get_config()
    return VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )


@st.cache_resource
def get_reranker():
    config = get_config()
    return Reranker(config.retrieval.rerank.model)


@st.cache_resource
def get_bm25_index():
    config = get_config()
    return BM25Index.from_vector_store(get_vector_store()) if config.retrieval.hybrid.enabled else None


@st.cache_resource
def get_doc_topics():
    config = get_config()
    return load_doc_topics(config.documents.metadata_file)


@st.cache_resource
def get_rag_pipeline():
    config = get_config()
    embedding_generator = EmbeddingGenerator(config.embedding.model_name)
    vector_store = get_vector_store()
    doc_topics = get_doc_topics()
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=config.retrieval.top_k,
        min_relevance_score=config.retrieval.min_relevance_score,
        bm25_index=get_bm25_index(),
        hybrid_candidates=config.retrieval.hybrid.candidates,
        reranker=get_reranker() if config.retrieval.rerank.enabled else None,
        rerank_candidates=config.retrieval.rerank.candidates,
        doc_topics=doc_topics,
    )
    llm_client = LlmClient(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        temperature=config.ollama.temperature,
        timeout_seconds=config.ollama.request_timeout_seconds,
        num_thread=config.ollama.num_thread,
        num_predict=config.ollama.num_predict,
    )
    return RagPipeline(retriever=retriever, llm_client=llm_client, doc_topics=doc_topics)


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


def render_sidebar(config, session_store):
    with st.sidebar:
        if DEBUG_ENABLED:
            st.caption("Debug mode: ON")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()


TOP_SOURCES_SHOWN = 2


def _citation_label(rank, source):
    """Formats a source as a numbered, research-paper-style citation,
    e.g. "[1] HoD Handbook - CMRIT, p. 20" or, for a chunk that spans a
    page break, "[1] HoD Handbook - CMRIT, p. 20-21".
    """
    title = Path(source.source_file).stem
    if source.page_end and source.page_end != source.page_number:
        page_label = f"p. {source.page_number}-{source.page_end}"
    else:
        page_label = f"p. {source.page_number}"
    return f"[{rank}] {title}, {page_label}"


def render_sources(sources, config, key_suffix, cited_indices=None):
    """Shows links to the source PDFs below the answer.

    When the answer text carries inline "[N]" citations (see
    rag_pipeline._normalize_citations), cited_indices pins each
    reference shown here to the exact N the answer body cited, from
    the application's own retrieval metadata — never text the LLM
    wrote — so the numbers always line up and nothing is fabricated.
    Falls back to the top TOP_SOURCES_SHOWN retrieved sources when
    nothing was cited (e.g. a short, non-committal reply).
    """
    if not sources:
        return
    if cited_indices:
        shown = [(index, sources[index - 1]) for index in cited_indices]
    else:
        shown = list(enumerate(sources[:TOP_SOURCES_SHOWN], start=1))

    # Different retrieved chunks can point at the same document/page
    # (e.g. two chunks off the same page, or a page-spanning chunk
    # that overlaps a single-page one), which would otherwise list an
    # identical citation more than once. Keep only the first rank seen
    # for each (source_file, page_number, page_end).
    seen_locations = set()
    deduped_shown = []
    for rank, source in shown:
        location = (source.source_file, source.page_number, source.page_end)
        if location in seen_locations:
            continue
        seen_locations.add(location)
        deduped_shown.append((rank, source))
    shown = deduped_shown

    st.caption("Sources:")
    for rank, source in shown:
        pdf_path = Path(config.documents.source_directory) / source.source_file
        citation = _citation_label(rank, source)
        download_column, link_column = st.columns([2, 3])

        with download_column:
            if pdf_path.exists():
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                st.download_button(
                    label=f"📄 {citation}",
                    data=pdf_bytes,
                    file_name=source.source_file,
                    mime="application/pdf",
                    key=f"source-{key_suffix}-{rank}",
                )
            else:
                st.caption(citation)

        with link_column:
            if source.source_link:
                st.caption(f"link: <{source.source_link}>")


def render_topic_link(topic_name, topic_link):
    """Shows the canonical URL for the topic docs-metadata.json says the
    question was about, if one was identified (see
    rag.docs_metadata.find_matching_topic). Shown regardless of whether
    the answer came from a locally-indexed chunk, since some topics
    (e.g. the college website) have no local PDF to index at all.
    """
    if not topic_link:
        return
    st.caption(f"🔗 Related: [{topic_name}]({topic_link})")


def render_estimated_response_time(session_store):
    """Shows the average total answer time over recent queries (timing
    stored on each assistant message — see SessionStore.add_message)
    as an estimate of how long the *next* answer will take. Rendered
    once on the main chat page, above the chat input, so it's visible
    before a question is even asked — see render_answer_time for the
    actual per-answer figure shown after each reply instead."""
    count, average_seconds = session_store.get_timing_summary()
    if average_seconds is None:
        return
    st.caption(f"⏱️ Estimated answer time (avg of last {count} questions): {average_seconds:.1f}s")


def render_answer_time(total_seconds):
    """Shows how long this specific answer took, right after it — the
    per-answer counterpart to the sidebar's running average."""
    if total_seconds is None:
        return
    st.caption(f"⏱️ Answered in {total_seconds:.1f}s")


# A handful of representative, already-indexed policy questions shown
# as one-click starting points, so a new user isn't stuck guessing
# what to type.
EXAMPLE_QUERIES = [
    "Give details on the leave application and approval process?",
    "Give details on the probation period completion process?",
    "What are the staff welfare measures at CMRIT?",
    "What is the CMRIT quality policy?",
    "What is the CMRIT grievance redressal policy?",
    "What is the CMRIT anti-ragging policy?",
]


def render_example_queries():
    """Shows EXAMPLE_QUERIES as clickable buttons in the sidebar, in
    place of typing. Rendered in the sidebar rather than the main chat
    page so it doesn't compete for space with the answer/citations,
    and kept visible for the whole session (not just before the first
    question) so it stays available as a quick shortcut throughout the
    conversation.
    """
    with st.sidebar:
        st.caption("Try one of these:")
        for example_query in EXAMPLE_QUERIES:
            if st.button(example_query, key=f"example-{example_query}", use_container_width=True):
                st.session_state.pending_question = example_query
                st.rerun()


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
            section_note = f" — section: {source.section_title}" if source.section_title else ""
            st.write(
                f"**{_citation_label(rank, source)}** "
                f"— similarity: {source.relevance_score:.3f} "
                f"— matched via: {source.match_type}"
                f"{section_note}"
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


def render_chat_page():
    config = get_config()
    session_store = get_session_store()
    ensure_active_session(session_store, config.session.timeout_minutes)

    render_page_title("CMRIT College Policy Assistant")
    st.write(
        "This assistant answers questions about CMRIT college policies "
        "using only the official policy documents. It will not answer "
        "questions unrelated to those policies."
    )
    st.caption(
        "Ask a question about CMRIT college policies. Answers are grounded "
        "in the official policy documents."
    )
    render_session_countdown(session_store, config)
    render_estimated_response_time(session_store)
    render_example_queries()

    for message_index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_sources(
                    message.get("sources", []),
                    config,
                    f"history-{message_index}",
                    cited_indices=message.get("cited_indices"),
                )
                render_topic_link(message.get("topic_name"), message.get("topic_link"))
                render_answer_time(message.get("total_seconds"))
                render_debug_panel(message.get("question", ""), message.get("sources", []))

    question = st.chat_input("Ask about a college policy...")
    is_new_question = question is not None
    if not question:
        # Set by render_example_queries() when the user clicks a
        # readymade question instead of typing one.
        question = st.session_state.pop("pending_question", None)
        is_new_question = question is not None
    if not question:
        # Nothing typed/clicked this run. If the last turn is a user
        # message with no assistant reply after it, the run that was
        # generating that reply got cut off mid-flight — e.g. the user
        # switched to the System Info page while the LLM was still
        # responding, which makes Streamlit abort the chat page's
        # script before it could append the answer. Resume answering
        # that question instead of leaving the chat stuck with no
        # visible progress and no way to continue.
        chat_history = st.session_state.chat_history
        if chat_history and chat_history[-1]["role"] == "user":
            question = chat_history[-1]["content"]

    if question:
        if is_new_question:
            # Captured before this turn is appended, so it holds only
            # earlier turns — the current question is sent separately.
            conversation_history = _build_conversation_history(
                st.session_state.chat_history, config.session.max_history_messages
            )
            session_store.add_message(st.session_state.session_id, "user", question)
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
        else:
            # Resuming: the question is already the last chat_history
            # entry (rendered by the loop above), so only the reply is
            # still owed.
            conversation_history = _build_conversation_history(
                st.session_state.chat_history[:-1], config.session.max_history_messages
            )

        with st.chat_message("assistant"):
            with st.spinner("Checking the policy documents..."):
                rag_pipeline = get_rag_pipeline()
                result = rag_pipeline.answer(question, conversation_history=conversation_history)
            # Idle countdown restarts from when the answer is delivered,
            # not from when the question was submitted, so a slow
            # generation doesn't eat into (or get cut short by) the
            # session's idle window.
            touch_session(session_store)
            st.write(result.answer_text)
            render_sources(
                result.sources,
                config,
                f"new-{len(st.session_state.chat_history)}",
                cited_indices=result.cited_indices,
            )
            render_topic_link(result.topic_name, result.topic_link)
            render_answer_time(result.total_seconds)
            render_debug_panel(question, result.sources)

        session_store.add_message(
            st.session_state.session_id,
            "assistant",
            result.answer_text,
            retrieval_seconds=result.retrieval_seconds,
            llm_seconds=result.llm_seconds,
            total_seconds=result.total_seconds,
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.answer_text,
                "sources": result.sources,
                "question": question,
                "topic_name": result.topic_name,
                "topic_link": result.topic_link,
                "cited_indices": result.cited_indices,
                "total_seconds": result.total_seconds,
            }
        )


def render_system_info():
    config = get_config()
    render_system_info_page(config, get_vector_store(), st.session_state.user_email)


def main():
    config = get_config()

    if "user_email" not in st.session_state:
        st.session_state.user_email = DEFAULT_USER_EMAIL

    render_sidebar(config, get_session_store())

    pages = st.navigation(
        [
            st.Page(render_chat_page, title="Chat", icon="💬", default=True),
            st.Page(render_system_info, title="System Info", icon="ℹ️"),
        ]
    )
    pages.run()
