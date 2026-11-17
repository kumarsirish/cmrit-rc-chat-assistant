"""Streamlit dashboard for this RAG app.

This is the only module that contains UI/dashboard code (besides
admin_page.py, the Admin tab's counterpart); every other module is a
plain Python component that this module calls into. `main.py` is the
entry point Streamlit runs (`streamlit run main.py`); it detects it's
running under Streamlit and calls this module's main() directly
instead of going through its own CLI.
"""

import json
import logging
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag._08_auth.session_store import SessionStore
from rag._08_auth.user_identity import normalize_email
from rag._09_common.config import PROJECT_ROOT, load_config
from rag._10_ui.admin_page import render_admin_page
from rag._10_ui.page_header import APP_ICON_PATH, render_centered_text, render_page_title
from rag._05_generation.llm_client import LlmClient
from rag._05_generation.rag_pipeline import RagPipeline
from rag._02_indexing.vector_store import VectorStore
from rag._01_ingestion.embeddings import EmbeddingGenerator
from rag._04_retrieval import faq_cache
from rag._04_retrieval.bm25_retriever import BM25Index
from rag._04_retrieval.dense_retriever import DenseRetriever
from rag._04_retrieval.reranker import Reranker
from rag._04_retrieval.retrieval_service import ALL_CATEGORIES, TRACE_LOGGER_NAME, RetrievalService
from rag._01_ingestion.document_loader import list_categories
from rag._01_ingestion.glossary_extractor import load_glossary
from rag._03_routing.category_matcher import EmbeddingCategoryMatcher
from rag._03_routing.docs_metadata import load_doc_topics

# Enabled by `make run-debug` (sets RAG_DEBUG=true). Shows the raw
# query, every retrieved chunk's source document/page, and its
# similarity score, so retrieval behaviour can be inspected live.
DEBUG_ENABLED = os.environ.get("RAG_DEBUG", "false").lower() == "true"

RETRIEVAL_DEBUG_LOG_PATH = PROJECT_ROOT / "logs" / "retrieval_debug.log"


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
def get_categories():
    config = get_config()
    return list_categories(config.documents.source_directory)


@st.cache_resource
def get_example_queries():
    """Returns up to EXAMPLE_QUERY_COUNT {"question", "category"} dicts
    for the sidebar's "Try one of these" buttons — one per distinct
    category (in file order), pulled from
    config.evaluation.test_data_path (tests/evaluation-question-
    answers.json) so the examples are always real, already-indexed
    questions rather than hand-maintained copies that can drift out of
    sync with the actual document set. The "unanswerable" category
    (see test_judge.py) is excluded — it's deliberately a refusal
    case, not a good "try this" example.

    Falls back to config.app.example_queries (plain strings, no
    category — clicking one leaves the category picker untouched) if
    the eval file is missing, empty, or has no categorized questions,
    so a deployment with no evaluation data configured still gets
    something in the sidebar.
    """
    config = get_config()
    try:
        with open(config.evaluation.test_data_path, "r", encoding="utf-8") as test_data_file:
            test_cases = json.load(test_data_file)
    except (OSError, json.JSONDecodeError):
        test_cases = []

    examples = []
    seen_categories = set()
    for case in test_cases:
        category = case.get("category")
        question = case.get("question")
        if not question or not category or category == "unanswerable" or category in seen_categories:
            continue
        seen_categories.add(category)
        examples.append({"question": question, "category": category})
        if len(examples) >= EXAMPLE_QUERY_COUNT:
            break

    if examples:
        return examples
    return [{"question": question, "category": None} for question in config.app.example_queries]


@st.cache_resource
def get_category_matcher():
    config = get_config()
    embedding_generator = EmbeddingGenerator(config.embedding.model_name)
    return EmbeddingCategoryMatcher(
        get_categories(), embedding_generator, similarity_threshold=config.retrieval.category_match_threshold
    )


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
        category_matcher=get_category_matcher(),
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
    config = get_config()
    return RagPipeline(
        retrieval_service=get_retrieval_service(),
        llm_client=get_llm_client(),
        glossary=get_glossary(),
        no_match_message=config.app.no_match_message,
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
    st.session_state.follow_up_count = 0


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
    st.session_state.follow_up_count = 0


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

# Chat page category picker's one special option (see render_chat_page)
# — maps to RetrievalService.retrieve()'s category_override semantics:
# "All categories" -> ALL_CATEGORIES; any other selection is a literal
# category name passed straight through (and, if that specific category
# comes back empty, retried as ALL_CATEGORIES — see render_chat_page).
ALL_CATEGORIES_LABEL = "All categories"

# Max consecutive questions within the same follow-up thread (see
# render_chat_page) — a question only counts toward this if
# RagAnswer.used_conversation_context says it actually depended on the
# preceding exchange; a self-contained/unrelated question doesn't
# count and resets the count to 0, starting a new thread. Once the
# limit is hit, conversation_history is silently dropped for the next
# question (not blocked, just answered as if it were a fresh one),
# which — since condensation then has nothing to depend on — makes
# that same question register as a fresh thread, naturally resetting
# the count.
FOLLOW_UP_LIMIT = 2

# Explicit key for the Chat page's category st.selectbox, so a sidebar
# example-question click (see render_example_queries) can set its
# value in code before the widget is (re)created on the next rerun —
# Streamlit reads st.session_state[key] to seed a widget's displayed
# value when that key is written before the widget runs.
CATEGORY_PICKER_KEY = "category_picker"

# How many example questions render_example_queries shows.
EXAMPLE_QUERY_COUNT = 4


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


def render_sources(sources, config, key_suffix, cited_indices=None) -> set:
    """Shows links to the source PDFs below the answer.

    When the answer text carries inline "[N]" citations (see
    rag_pipeline._normalize_citations), cited_indices pins each
    reference shown here to the exact N the answer body cited, from
    the application's own retrieval metadata — never text the LLM
    wrote — so the numbers always line up and nothing is fabricated.
    Falls back to the top TOP_SOURCES_SHOWN retrieved sources when
    nothing was cited (e.g. a short, non-committal reply).

    Returns the set of source_link values actually shown, so a caller
    (see render_chat_page) can skip render_topic_link's "Related:"
    line when it would just repeat a link already shown here.
    """
    if not sources:
        return set()
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
    shown_source_links = set()
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
                # Same "name: link" presentation as render_topic_link's
                # "Related:" line (document title as the hyperlink text,
                # not the bare URL) — kept consistent between the two.
                title = Path(source.source_file).stem
                st.caption(f"[{title}]({source.source_link})")
                shown_source_links.add(source.source_link)

    return shown_source_links


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
        "_(This answer is of low-confidence — please verify against the source document directly.)_"
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
        # Marks this retry so render_chat_page tells the pipeline not to
        # offer yet another rephrase suggestion if this one also fails —
        # see RagPipeline.answer's suggest_rephrase parameter.
        st.session_state.pending_question_is_rephrase_retry = True
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


def render_rating(session_store, message_id, current_rating, widget_key):
    """Shows a 1-5 star rating widget under an answer and persists a
    change to the database (see SessionStore.rate_message).

    message_id is the assistant message's row id in MySQL (see
    SessionStore.add_message's return value) — None for a message that
    predates this feature (nothing to rate_message() against), in
    which case this renders nothing.

    widget_key must be derived from message_id alone (not a "new-"/
    "history-" prefix the way render_sources's key_suffix is) — same
    reasoning as render_rephrase_suggestion's key: clicking a star
    triggers a rerun on which this message has already moved from the
    "new answer" render path to the "history replay" one, and
    Streamlit only keeps a widget's interaction if a widget with the
    exact same key is recreated on that next rerun.

    st.feedback's own widget state (keyed by widget_key) already
    survives reruns, so current_rating/default only matters for
    restoring the right selection on a fresh page load (e.g. after
    switching tabs and back) — rate_message() is still only called
    when the selection actually changes, not on every rerun.
    """
    if message_id is None:
        return
    selected = st.feedback(
        "stars",
        key=widget_key,
        default=(current_rating - 1) if current_rating else None,
    )
    if selected is not None and selected + 1 != current_rating:
        session_store.rate_message(message_id, selected + 1)


def render_example_queries(example_queries):
    """Shows example_queries (see get_example_queries — a handful of
    representative, already-indexed questions, each tagged with the
    category it belongs to) as clickable buttons in the sidebar, in
    place of typing. Rendered in the sidebar rather than the main chat
    page so it doesn't compete for space with the answer/citations,
    and kept visible for the whole session (not just before the first
    question) so it stays available as a quick shortcut throughout the
    conversation.

    Clicking one sets both the question AND the category picker
    (st.session_state[CATEGORY_PICKER_KEY]) to that question's own
    category — read the next rerun before the selectbox is (re)created,
    so the question is asked exactly as scoped as it was written for,
    not whatever category happened to be selected before. A fallback
    example with no category (category is None — see
    get_example_queries) leaves the picker untouched instead.
    """
    if not example_queries:
        return
    with st.sidebar:
        st.caption("Try one of these:")
        for example in example_queries:
            question = example["question"]
            if st.button(question, key=f"example-{question}", use_container_width=True):
                st.session_state.pending_question = question
                if example["category"] is not None:
                    st.session_state[CATEGORY_PICKER_KEY] = example["category"]
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

    render_page_title(config.app.name)
    render_centered_text(config.app.description)
    render_session_countdown(session_store, config)
    render_estimated_response_time(session_store)
    render_example_queries(get_example_queries())

    for message_index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_low_confidence_warning(message.get("low_confidence", False))
                shown_source_links = render_sources(
                    message.get("sources", []),
                    config,
                    f"history-{message_index}",
                    cited_indices=message.get("cited_indices"),
                )
                if message.get("topic_link") not in shown_source_links:
                    render_topic_link(message.get("topic_name"), message.get("topic_link"))
                render_rephrase_suggestion(message.get("rephrase_suggestion"), message_index)
                render_answer_time(message.get("total_seconds"))
                render_rating(
                    session_store,
                    message.get("message_id"),
                    message.get("rating"),
                    widget_key=f"rating-{message.get('message_id')}",
                )
                render_debug_panel(message.get("question", ""), message.get("sources", []))

    category_option = st.selectbox(
        "Category",
        [ALL_CATEGORIES_LABEL] + get_categories(),
        key=CATEGORY_PICKER_KEY,
        help=(
            "Pick a specific category to search only it first — if nothing "
            "is found there, every other category is searched automatically. "
            "'All categories' always searches everything directly."
        ),
    )
    category_override = ALL_CATEGORIES if category_option == ALL_CATEGORIES_LABEL else category_option

    question = st.chat_input("Ask a question about the indexed documents...")
    is_new_question = question is not None
    is_rephrase_retry = False
    if not question:
        # Set by render_example_queries() (or render_rephrase_suggestion)
        # when the user clicks a readymade question instead of typing one.
        question = st.session_state.pop("pending_question", None)
        is_new_question = question is not None
        # Only set when the readymade question was itself a rephrase
        # suggestion (see render_rephrase_suggestion) — tells
        # rag_pipeline.answer not to offer yet another one if this
        # retry also fails.
        is_rephrase_retry = st.session_state.pop("pending_question_is_rephrase_retry", False)
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

        # Once FOLLOW_UP_LIMIT consecutive context-dependent follow-ups
        # have been asked, silently drop conversation_history for this
        # turn — the question is still answered, just without the
        # preceding exchange to depend on. Since query_condenser then
        # has nothing to condense against, this turn's
        # used_conversation_context will come back False, which is
        # exactly what resets follow_up_count to 0 below — a new
        # follow-up thread can then build back up from scratch.
        effective_conversation_history = (
            None if st.session_state.get("follow_up_count", 0) >= FOLLOW_UP_LIMIT else conversation_history
        )

        with st.chat_message("assistant"):
            rag_pipeline = get_rag_pipeline()
            spinner_label = (
                "Checking the indexed documents..."
                if category_override == ALL_CATEGORIES
                else f"Checking '{category_override}'..."
            )
            with st.spinner(spinner_label):
                result = rag_pipeline.answer(
                    question,
                    conversation_history=effective_conversation_history,
                    suggest_rephrase=not is_rephrase_retry,
                    category_override=category_override,
                )
            # A specific category (not ALL_CATEGORIES) came back empty —
            # widen the search to every other category rather than
            # stopping at "not found in this one category", discarding
            # this first (unanswered) attempt entirely in favor of
            # whatever the broader search finds.
            if category_override != ALL_CATEGORIES and not result.grounded:
                st.info(f"No answer found in '{category_override}'. Checking other categories...")
                with st.spinner("Checking other categories..."):
                    result = rag_pipeline.answer(
                        question,
                        conversation_history=effective_conversation_history,
                        suggest_rephrase=not is_rephrase_retry,
                        category_override=ALL_CATEGORIES,
                    )
            st.session_state.follow_up_count = (
                st.session_state.get("follow_up_count", 0) + 1 if result.used_conversation_context else 0
            )
            # Idle countdown restarts from when the answer is delivered,
            # not from when the question was submitted, so a slow
            # generation doesn't eat into (or get cut short by) the
            # session's idle window.
            touch_session(session_store)
            st.write(result.answer_text)
            render_low_confidence_warning(result.low_confidence)
            shown_source_links = render_sources(
                result.sources,
                config,
                f"new-{len(st.session_state.chat_history)}",
                cited_indices=result.cited_indices,
            )
            if result.topic_link not in shown_source_links:
                render_topic_link(result.topic_name, result.topic_link)
            render_rephrase_suggestion(result.rephrase_suggestion, len(st.session_state.chat_history))
            render_answer_time(result.total_seconds)
            message_id = session_store.add_message(
                st.session_state.session_id,
                "assistant",
                result.answer_text,
                retrieval_seconds=result.retrieval_seconds,
                llm_seconds=result.llm_seconds,
                total_seconds=result.total_seconds,
            )
            render_rating(
                session_store, message_id, current_rating=None, widget_key=f"rating-{message_id}"
            )
            render_debug_panel(question, result.sources)

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
                "message_id": message_id,
                "rating": None,
            }
        )


def render_admin():
    # System Info and Recent Queries moved here as tabs (see
    # admin_page.render_admin_page) instead of their own top-level
    # pages, so the sidebar nav only lists Chat and Admin.
    render_admin_page(get_config(), st.session_state.user_email)


def main():
    config = get_config()

    # st.set_page_config must run before any other Streamlit call in
    # this script run, and must run here (inside main(), executed on
    # every session) rather than at module import time: a module's
    # top-level code only runs once per server process (Python caches
    # the import), so a module-level call here would only ever reach
    # the first browser session that process serves — every later
    # session would fall through to Streamlit's literal default tab
    # title "Streamlit" instead of config.app.name.
    st.set_page_config(
        page_title=config.app.name,
        page_icon=str(APP_ICON_PATH),
        layout="wide",
        initial_sidebar_state="collapsed",
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
        .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 100% !important;
            padding-top: 2.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin: 0 !important;
        }
        [data-testid="stSidebar"] .block-container,
        [data-testid="stSidebar"] [data-testid="stMainBlockContainer"] {
            padding-top: 1.25rem !important;
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

    if "user_email" not in st.session_state:
        st.session_state.user_email = normalize_email(config.app.default_user_email)

    render_sidebar(config, get_session_store())

    pages = st.navigation(
        [
            st.Page(render_chat_page, title="Chat", icon="💬", default=True),
            st.Page(render_admin, title="Admin", icon="🔧"),
        ]
    )
    pages.run()
