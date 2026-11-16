"""Streamlit dashboard for the CMRIT college policy assistant.

This is the only module that contains UI/dashboard code; every other
module is a plain Python component that this module calls into.
`main.py` is the entry point Streamlit runs (`streamlit run main.py`);
it detects it's running under Streamlit and calls this module's
main() directly instead of going through its own CLI.
"""

import logging
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag._08_auth.session_store import SessionStore
from rag._08_auth.user_identity import normalize_email
from rag._09_common.config import PROJECT_ROOT, load_config
from rag._09_common.system_info import render_system_info_page
from rag._05_generation.llm_client import LlmClient
from rag._05_generation.rag_pipeline import RagPipeline
from rag._02_indexing.vector_store import VectorStore
from rag._01_ingestion.embeddings import EmbeddingGenerator
from rag._04_retrieval import faq_cache
from rag._04_retrieval.bm25_retriever import BM25Index
from rag._04_retrieval.dense_retriever import DenseRetriever
from rag._04_retrieval.reranker import Reranker
from rag._04_retrieval.retrieval_service import TRACE_LOGGER_NAME, RetrievalService
from rag._01_ingestion.glossary_extractor import load_glossary
from rag._03_routing.docs_metadata import load_doc_topics

APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.png"

st.set_page_config(
    page_title="CMRIT-RAG", page_icon=str(APP_ICON_PATH), layout="wide", initial_sidebar_state="collapsed"
)

# Streamlit's default centered layout leaves large blank margins on
# either side of the chat, plus a tall gap above the title. Constrain
# the content to a comfortable reading width and pull it up/left
# closer to the edge of the viewport instead.
#
# Streamlit's own header toolbar (hamburger menu / "Deploy" button) is
# position:fixed at the top of the viewport, normally cleared by
# block-container's large default padding-top. Shrinking that padding
# without also shrinking the header leaves the header's fixed height
# taller than the new clearance, so content (the page-title icon in
# particular, being an <img> rather than text) renders partly behind
# it and looks clipped at the top. Shrinking the header to match keeps
# the compact layout without that overlap.
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        height: 2.5rem;
        min-height: 2.5rem;
    }
    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
        margin: 0;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 0.75rem;
        padding: 0.5rem 0.25rem;
    }
    hr {
        margin: 0.75rem 0;
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
def get_retrieval_service():
    config = get_config()
    embedding_generator = EmbeddingGenerator(config.embedding.model_name)
    dense_retriever = DenseRetriever(get_vector_store())
    doc_topics = get_doc_topics()
    retrieval_service = RetrievalService(
        dense_retriever=dense_retriever,
        embedding_generator=embedding_generator,
        top_k=config.retrieval.top_k,
        min_reranker_score=config.retrieval.min_reranker_score,
        min_relevance_score=config.retrieval.min_relevance_score,
        bm25_index=get_bm25_index(),
        hybrid_candidates=config.retrieval.hybrid.candidates,
        dense_weight=config.retrieval.hybrid.dense_weight,
        lexical_weight=config.retrieval.hybrid.lexical_weight,
        reranker=get_reranker() if config.retrieval.rerank.enabled else None,
        rerank_candidates=config.retrieval.rerank.candidates,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=config.retrieval.mapped_document_primary_top_k,
        topic_match_threshold=config.retrieval.topic_match_threshold,
        low_confidence_min_score=config.retrieval.low_confidence_min_score,
    )
    return faq_cache.maybe_wrap(retrieval_service, config.redis)


@st.cache_resource
def get_llm_client():
    config = get_config()
    return LlmClient.from_config(config.llm, temperature=_default_llm_temperature(config))


@st.cache_resource
def get_glossary():
    config = get_config()
    return load_glossary(config.documents.glossary_file)


def get_rag_pipeline():
    return RagPipeline(
        retrieval_service=get_retrieval_service(),
        llm_client=get_llm_client(),
        glossary=get_glossary(),
    )


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
    st.session_state.history_reset_at_index = 0


def touch_session(session_store):
    session_store.touch_session(st.session_state.session_id)
    st.session_state.last_active_at = datetime.now()


def reset_expired_session(session_store, user_email):
    """Starts a fresh backend session after a timeout, without wiping
    the chat already on screen. Only the session/memory (session_id,
    idle timer, and the conversation history sent to the LLM going
    forward) resets; previously displayed messages stay visible.

    history_reset_at_index marks where the reset happened, so
    _build_conversation_history (the context sent to the LLM) only
    looks at messages from after this point — the messages before it
    are still shown on screen, just no longer treated as "memory".
    """
    session_id = session_store.create_session(user_email)
    st.session_state.session_id = session_id
    st.session_state.last_active_at = datetime.now()
    st.session_state.history_reset_at_index = len(st.session_state.chat_history)


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
        reset_expired_session(session_store, user_email)


@st.fragment(run_every=1)
def render_session_countdown(session_store, config):
    """Live, self-updating idle countdown.

    Runs on its own 1-second timer (via st.fragment) independent of
    user interaction. Once idle time crosses the warning threshold it
    shows a countdown; once idle time reaches the full timeout it
    resets the session/memory and reruns the app, without needing the
    user to do anything first. The chat already on screen stays put —
    only the session memory (and the conversation history sent to the
    LLM for the next question) resets, see reset_expired_session.
    """
    timeout_seconds = config.session.timeout_minutes * 60
    warning_seconds = config.session.warning_seconds_before_timeout
    idle_seconds = (datetime.now() - st.session_state.last_active_at).total_seconds()
    remaining_seconds = timeout_seconds - idle_seconds

    if remaining_seconds <= 0:
        reset_expired_session(session_store, st.session_state.user_email)
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


def _default_llm_temperature(config):
    provider_config = getattr(config.llm, config.llm.provider)
    # config.yaml can write this as a plain "0", which YAML parses as
    # an int rather than a float.
    return float(provider_config.temperature)


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

    # Different retrieved chunks can point at the same source document
    # (e.g. two chunks off the same page, or adjacent/overlapping page
    # ranges from splitting that document into chunks), which would
    # otherwise list the same file more than once. Merge all chunks
    # from the same source_file into a single citation, using the
    # first rank seen and a page range spanning every page any of its
    # chunks covered.
    merged_by_file = {}
    file_order = []
    for rank, source in shown:
        entry = merged_by_file.get(source.source_file)
        if entry is None:
            merged_by_file[source.source_file] = {
                "rank": rank,
                "source": source,
                "page_start": source.page_number,
                "page_end": source.page_end,
            }
            file_order.append(source.source_file)
        else:
            entry["page_start"] = min(entry["page_start"], source.page_number)
            entry["page_end"] = max(entry["page_end"], source.page_end)

    shown = []
    for source_file in file_order:
        entry = merged_by_file[source_file]
        merged_source = replace(
            entry["source"], page_number=entry["page_start"], page_end=entry["page_end"]
        )
        shown.append((entry["rank"], merged_source))

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


def render_low_confidence_warning(low_confidence: bool) -> None:
    """Shown whenever the answer relies on the retrieval_service
    low-confidence fallback (see RagAnswer.low_confidence) — nothing
    cleared the normal relevance threshold, so the single closest
    match was used anyway. This is the only place the caveat is
    shown — rag_pipeline.SYSTEM_PROMPT explicitly tells the LLM not
    to add its own, so it isn't duplicated in the answer text (and
    doesn't show up as an ungrounded claim when RAGAS scores
    faithfulness on the answer).
    """
    if not low_confidence:
        return
    st.caption(
        "_(This match was low-confidence — please verify against the "
        "source document directly.)_"
    )


def render_rephrase_suggestion(rephrase_suggestion, message_index):
    """Shown whenever the pipeline couldn't find anything for the plain
    "no information found" refusal (see
    rag._05_generation.rephrase_suggester) — nudges the user to
    reframe their question and offers a concrete HyDE-style
    alternative phrasing as a one-click retry, instead of leaving them
    to guess a better wording themselves.

    message_index must be this message's *eventual* index in
    chat_history — i.e. len(chat_history) at the moment it's answered
    (before it's appended), which is the same value message_index
    takes when that same message is later replayed from history. The
    button's key depends on it (not a "new"/"history" prefix like
    render_sources uses) because, unlike a plain download button, this
    one has a side effect: clicking it sets pending_question and calls
    st.rerun(). Streamlit only honors a click if a widget with the
    exact same key is created again in the very next rerun — and that
    next rerun is exactly when this message moves from the "new answer"
    render path to the "history replay" one. A "new-"/"history-"
    prefix would change the key across that transition and silently
    swallow the click.
    """
    if not rephrase_suggestion:
        return
    st.info("Try reframing your question. For example:")
    if st.button(rephrase_suggestion, key=f"rephrase-{message_index}", use_container_width=True):
        st.session_state.pending_question = rephrase_suggestion
        st.rerun()


def render_topic_link(topic_name, topic_link):
    """Shows the canonical URL for the topic docs-metadata.json says the
    question was about, if one was identified (see
    rag._03_routing.topic_matcher.EmbeddingTopicMatcher). Shown regardless of whether
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
# what to type. Limited to questions verified to score RAGAS
# faithfulness == 1.0 against the live pipeline — "grievance
# redressal" and "anti-ragging" were dropped because the pipeline
# currently can't answer them at all (no matching indexed document),
# so clicking them would only produce a refusal.
EXAMPLE_QUERIES = [
    "Give details on the leave application and approval process?",
    "Give details on the probation period completion process?",
    "What are the staff welfare measures at CMRIT?",
    "What is the CMRIT quality policy?",
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


def _build_conversation_history(chat_history, max_messages, min_index):
    """Turns the session's chat_history so far into the {role, content}
    list the LLM expects, capped to the most recent max_messages turns
    so a long session doesn't keep growing the prompt indefinitely.

    min_index excludes anything before the last session reset (see
    reset_expired_session) — those messages are still shown on screen,
    but a timed-out session's memory shouldn't carry into new answers.
    """
    messages_since_reset = chat_history[min_index:]
    recent_messages = messages_since_reset[-max_messages:]
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
                render_low_confidence_warning(message.get("low_confidence", False))
                render_sources(
                    message.get("sources", []),
                    config,
                    f"history-{message_index}",
                    cited_indices=message.get("cited_indices"),
                )
                render_topic_link(message.get("topic_name"), message.get("topic_link"))
                render_rephrase_suggestion(message.get("rephrase_suggestion"), message_index)
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
                st.session_state.chat_history,
                config.session.max_history_messages,
                st.session_state.history_reset_at_index,
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
                st.session_state.chat_history[:-1],
                config.session.max_history_messages,
                st.session_state.history_reset_at_index,
            )

        with st.chat_message("assistant"):
            with st.spinner("Checking the policy documents..."):
                rag_pipeline = get_rag_pipeline()
                result = rag_pipeline.answer(
                    question,
                    conversation_history=conversation_history,
                )
            # Idle countdown restarts from when the answer is delivered,
            # not from when the question was submitted, so a slow
            # generation doesn't eat into (or get cut short by) the
            # session's idle window.
            touch_session(session_store)
            st.write(result.answer_text)
            render_low_confidence_warning(result.low_confidence)
            render_sources(
                result.sources,
                config,
                f"new-{len(st.session_state.chat_history)}",
                cited_indices=result.cited_indices,
            )
            render_topic_link(result.topic_name, result.topic_link)
            render_rephrase_suggestion(result.rephrase_suggestion, len(st.session_state.chat_history))
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
                "low_confidence": result.low_confidence,
                "rephrase_suggestion": result.rephrase_suggestion,
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
