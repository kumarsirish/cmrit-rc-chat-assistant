"""System Info page: read-only overview of what's currently running.

Shows the models in use, the configured RAGAS judge setup, the source
PDFs the assistant was ingested from, and when the vector store was
last (re)built — useful for confirming the deployed config matches
what's expected without reading config.yaml directly.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from rag.evaluation.baseline import load_baseline_mean_scores
from rag.ingestion.document_loader import find_pdf_files

CHROMA_SQLITE_FILENAME = "chroma.sqlite3"

IST = ZoneInfo("Asia/Kolkata")


def _format_ist(dt: datetime) -> str:
    """Formats a datetime as IST, e.g. '2026-09-08 07:30:29 IST'.

    Naive datetimes (e.g. from datetime.fromtimestamp) are assumed to
    already be in local system time; aware datetimes (e.g. a parsed
    UTC ISO string) are converted from their own timezone.
    """
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _format_timestamp_ist(unix_timestamp: float) -> str:
    return _format_ist(datetime.fromtimestamp(unix_timestamp, tz=timezone.utc))

# One-line, plain-English explanation of each RAGAS metric, shown next
# to its score so the number means something without reading the
# RAGAS docs. Kept short — the formula (METRIC_FORMULAS) carries the
# precise definition on its own line underneath.
METRIC_DESCRIPTIONS = {
    "faithfulness": "Fraction of the answer's claims actually backed by the retrieved context (higher = less hallucination).",
    "answer_relevancy": "How directly the answer addresses the question that was actually asked.",
    "answer_correctness": "How closely the answer matches the reference (ground-truth) answer.",
    "context_precision": "How much of the retrieved context was actually relevant (little irrelevant noise retrieved).",
    "context_recall": "How much of the information needed to answer was present in the retrieved context at all.",
}

# LaTeX (rendered via KaTeX in st.markdown's `$...$`), one formula per
# metric — the precise textbook definition, shown on its own line
# under each one-liner above. `\lvert x \rvert` instead of literal
# `|x|`: a bare `|` inside a markdown table cell is parsed as a column
# delimiter and breaks the table.
METRIC_FORMULAS = {
    "faithfulness": r"$\text{Faithfulness} = \dfrac{\lvert\text{supported claims}\rvert}{\lvert\text{claims in answer}\rvert}$",
    "answer_relevancy": r"$\text{Answer Relevancy} = \dfrac{1}{n}\sum_{i=1}^{n}\cos(E_{g_i}, E_o)$",
    "answer_correctness": r"$\text{Answer Correctness} = w_1 F_1 + w_2\cos(a, r),\ F_1=\dfrac{TP}{TP+0.5(FP+FN)}$",
    "context_precision": r"$\text{Context Precision@K} = \dfrac{\sum_{k=1}^{K}\text{Precision@k}\times rel_k}{\lvert\text{relevant in top }K\rvert}$",
    "context_recall": r"$\text{Context Recall} = \dfrac{\lvert\text{GT statements attributable to context}\rvert}{\lvert\text{GT statements}\rvert}$",
}

# Same idea, for the classic IR metrics in tests/evaluation/
# test_retrieval_eval.py (rag/evaluation/retrieval_metrics.py) — scores
# retrieval alone, with no LLM call involved.
RETRIEVAL_METRIC_LABELS = {
    "recall": "Recall",
    "precision": "Precision",
    "reciprocal_rank": "MRR",
    "ndcg": "NDCG",
}
RETRIEVAL_METRIC_DESCRIPTIONS = {
    "recall": "Fraction of the question's known-relevant documents that were actually retrieved.",
    "precision": "Fraction of the retrieved documents that were actually relevant.",
    "reciprocal_rank": "How close to the top the first relevant document was ranked, averaged over questions.",
    "ndcg": "Rewards relevant documents ranked higher, not just present.",
}
RETRIEVAL_METRIC_FORMULAS = {
    "recall": r"$\text{Recall@k} = \dfrac{\lvert Rel \cap Ret_k\rvert}{\lvert Rel\rvert}$",
    "precision": r"$\text{Precision@k} = \dfrac{\lvert Rel \cap Ret_k\rvert}{k}$",
    "reciprocal_rank": r"$RR=\dfrac{1}{\text{rank}_{\text{first relevant}}},\ MRR=\dfrac{1}{\lvert Q\rvert}\sum_{q\in Q}RR_q$",
    "ndcg": r"$NDCG@k=\dfrac{DCG@k}{IDCG@k},\ DCG@k=\sum_{i=1}^{k}\dfrac{rel_i}{\log_2(i+1)}$",
}


def _format_score(score) -> str:
    """"—" for missing/NaN (a NaN mean score means a judge call timed
    out/errored on the last run — see test_ragas_judge.py), otherwise
    3 decimal places."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "—"
    return f"{score:.3f}"


def _metric_table_rows(
    metric_names: list,
    descriptions: dict,
    formulas: dict,
    labels: dict,
    mean_scores: dict,
    baseline_scores: dict,
    min_score: float = None,
) -> list:
    """Builds the {Metric, What it means, Baseline, Last run} rows
    shared by the RAGAS and retrieval-eval sections. "What it means"
    is the one-liner plus its formula on a following line (`<br>` —
    plain "\\n" doesn't break within a markdown table cell). labels
    maps a raw metric key to its display name (e.g. "reciprocal_rank"
    -> "MRR"); pass {} to use the raw key as-is. min_score, if given,
    prefixes "Last run" with a pass/fail mark instead of just the
    number.
    """
    rows = []
    for name in metric_names:
        score = mean_scores.get(name) if mean_scores else None
        baseline = baseline_scores.get(name) if baseline_scores else None
        last_run_text = _format_score(score)
        if score is not None and not (isinstance(score, float) and math.isnan(score)) and min_score is not None:
            last_run_text = f"{'✅' if score >= min_score else '❌'} {last_run_text}"
        what_it_means = descriptions.get(name, "")
        formula = formulas.get(name, "")
        if formula:
            what_it_means = f"{what_it_means}<br>{formula}"
        rows.append(
            {
                "Metric": labels.get(name, name),
                "What it means": what_it_means,
                "Baseline": _format_score(baseline),
                "Last run": last_run_text,
            }
        )
    return rows


def _render_metric_table(rows: list) -> None:
    """Renders rows (from _metric_table_rows) as a markdown table
    instead of st.table: st.table can't render LaTeX or a `<br>` line
    break inside a cell, while st.markdown's table support does (both
    go through the same markdown+KaTeX pipeline). The "What it means"
    column is capped narrower than the others via inline width/wrap
    styling on its cell content, since a plain markdown table has no
    per-column width syntax of its own.
    """
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join('---' for _ in headers)} |"]
    for row in rows:
        cells = []
        for header in headers:
            value = str(row[header])
            if header == "What it means":
                value = f'<span style="display:inline-block;max-width:260px;white-space:normal;">{value}</span>'
            cells.append(value)
        lines.append(f"| {' | '.join(cells)} |")
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def _format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _render_models_section(config):
    st.subheader("Models")
    provider = config.llm.provider
    if provider == "ollama":
        answer_llm_line = (
            f"- **Answer LLM:** `{config.llm.ollama.model}` "
            f"(temperature={config.llm.ollama.temperature}, via Ollama at "
            f"`{config.llm.ollama.base_url}`)\n"
        )
    else:
        provider_cfg = getattr(config.llm, provider)
        answer_llm_line = (
            f"- **Answer LLM:** `{provider_cfg.model}` "
            f"(temperature={provider_cfg.temperature}, via "
            f"{provider.capitalize()})\n"
        )
    st.markdown(
        answer_llm_line
        + f"- **Embedding model:** `{config.embedding.model_name}`\n"
        f"- **RAGAS judge model:** `{config.evaluation.judge_model}` "
        f"(temperature={config.evaluation.judge_temperature}) — deliberately "
        f"a different model than the answer LLM, so it isn't grading its "
        f"own generations\n"
        f"- **Retrieval:** "
        + (
            f"hybrid (dense + BM25, {config.retrieval.hybrid.candidates} "
            f"candidates each, fused by reciprocal rank)"
            if config.retrieval.hybrid.enabled
            else "dense only"
        )
        + "\n"
        f"- **Reranker:** "
        + (
            f"`{config.retrieval.rerank.model}` (reranks "
            f"{config.retrieval.rerank.candidates} fused candidates, keeps top "
            f"{config.retrieval.top_k})"
            if config.retrieval.rerank.enabled
            else "disabled"
        )
    )


def _render_ragas_section(config):
    st.subheader("RAGAS evaluation")
    st.markdown(
        f"- **Metrics:** {', '.join(config.evaluation.metrics)}\n"
        f"- **Pass threshold:** mean score ≥ {config.evaluation.min_score} "
        f"per metric\n"
        f"- **Judge provider:** `{config.evaluation.judge_provider}`\n"
        f"- **Judge model:** `{config.evaluation.judge_model}`"
    )

    results_path = Path(config.evaluation.results_path)
    results = None
    if not results_path.exists():
        st.caption(
            f"No results yet — run `make test-ragas` to generate "
            f"`{results_path.name}`."
        )
    else:
        try:
            with open(results_path, "r", encoding="utf-8") as results_file:
                results = json.load(results_file)
            run_at = _format_ist(datetime.fromisoformat(results["run_at"]))
            st.markdown(f"**Last run:** {run_at} (judge: `{results['judge_model']}`)")
        except (json.JSONDecodeError, OSError, KeyError) as error:
            st.caption(f"Could not read {results_path.name}: {error}")

    try:
        baseline_scores = load_baseline_mean_scores(config.evaluation.baseline_path)
    except (json.JSONDecodeError, OSError, KeyError):
        baseline_scores = None

    _render_metric_table(
        _metric_table_rows(
            config.evaluation.metrics,
            METRIC_DESCRIPTIONS,
            METRIC_FORMULAS,
            {},
            results["mean_scores"] if results else None,
            baseline_scores,
            min_score=config.evaluation.min_score,
        )
    )
    if baseline_scores is None:
        st.caption(
            "No baseline saved yet — run `make save-ragas-baseline` to "
            "compare future runs against this one."
        )

    test_data_path = Path(config.evaluation.test_data_path)
    if not test_data_path.exists():
        st.caption(f"Reference Q&A file not found: {test_data_path}")
        return
    try:
        with open(test_data_path, "r", encoding="utf-8") as test_file:
            test_cases = json.load(test_file, strict=False)
        st.caption(
            f"{len(test_cases)} reference Q&A pairs in `{test_data_path.name}`. "
            "Run `make test-ragas` to score the live pipeline against them."
        )
    except (json.JSONDecodeError, OSError) as error:
        st.caption(f"Could not read {test_data_path.name}: {error}")


def _render_retrieval_eval_section(config):
    st.subheader("Retrieval evaluation")
    st.markdown(
        f"- **Metrics:** Recall, Precision, MRR, NDCG @ top_k={config.retrieval.top_k}\n"
        f"- **Pass thresholds:** Recall ≥ {config.evaluation.retrieval.min_recall}, "
        f"MRR ≥ {config.evaluation.retrieval.min_mrr}"
    )

    results_path = Path(config.evaluation.retrieval.results_path)
    results = None
    if not results_path.exists():
        st.caption(
            f"No results yet — run `make test-retrieval-eval` to generate "
            f"`{results_path.name}`."
        )
    else:
        try:
            with open(results_path, "r", encoding="utf-8") as results_file:
                results = json.load(results_file)
        except (json.JSONDecodeError, OSError, KeyError) as error:
            st.caption(f"Could not read {results_path.name}: {error}")

    try:
        baseline_scores = load_baseline_mean_scores(config.evaluation.retrieval.baseline_path)
    except (json.JSONDecodeError, OSError, KeyError):
        baseline_scores = None

    _render_metric_table(
        _metric_table_rows(
            RETRIEVAL_METRIC_LABELS.keys(),
            RETRIEVAL_METRIC_DESCRIPTIONS,
            RETRIEVAL_METRIC_FORMULAS,
            RETRIEVAL_METRIC_LABELS,
            results["mean_scores"] if results else None,
            baseline_scores,
        )
    )
    if baseline_scores is None:
        st.caption(
            "No baseline saved yet — run `make save-retrieval-baseline` to "
            "compare future runs against this one."
        )


def _render_documents_section(config, vector_store):
    st.subheader("Source documents")
    try:
        pdf_paths = find_pdf_files(config.documents.source_directory)
    except FileNotFoundError:
        st.caption(f"Documents directory not found: {config.documents.source_directory}")
        return

    if not pdf_paths:
        st.caption(f"No PDFs found in {config.documents.source_directory}")
        return

    for pdf_path in pdf_paths:
        stat = pdf_path.stat()
        modified = _format_timestamp_ist(stat.st_mtime)
        st.markdown(f"- {pdf_path.name} — {_format_bytes(stat.st_size)}, modified {modified}")

    st.caption(
        f"{len(pdf_paths)} document(s) — "
        f"{vector_store.count()} chunks currently in the "
        f"`{config.chroma.collection_name}` vector store collection"
    )


def _render_session_section(config, user_email):
    st.subheader("Session")
    st.markdown(
        f"- **Signed in as:** {user_email}\n"
        f"- **Session timeout:** {config.session.timeout_minutes} minutes\n"
        f"- **Timeout warning:** {config.session.warning_seconds_before_timeout} "
        f"seconds before timeout\n"
        f"- **Chat history sent to LLM:** last "
        f"{config.session.max_history_messages} messages"
    )


def _render_ingestion_section(config):
    st.subheader("Last ingestion")
    sqlite_path = Path(config.chroma.persist_directory) / CHROMA_SQLITE_FILENAME
    if not sqlite_path.exists():
        st.caption(
            "No vector store found yet — run `make ingest` to build the index "
            "from the source documents."
        )
        return

    last_modified = _format_timestamp_ist(sqlite_path.stat().st_mtime)
    st.caption(
        f"Vector store last written: {last_modified} "
        f"(`{config.chroma.persist_directory}`). Re-run `make ingest` after "
        "changing the PDFs in policies_docs."
    )


def render_system_info_page(config, vector_store, user_email):
    st.title("System Info")
    st.caption("Read-only overview of the models, evaluation setup, and documents in use.")

    _render_models_section(config)
    st.divider()
    _render_retrieval_eval_section(config)
    st.divider()
    _render_ragas_section(config)
    st.divider()
    _render_session_section(config, user_email)
    st.divider()
    _render_documents_section(config, vector_store)
    st.divider()
    _render_ingestion_section(config)
