"""DeepEval-judged evaluation of the live RAG pipeline.

For every {question, answer} pair in the file at
config.evaluation.test_data_path, runs the real retriever + LLM
pipeline to get an actual answer and its retrieved context, then asks
a separate judge model (config.evaluation.judge_model, e.g.
gemini-3.2-flash-lite — deliberately not the same model that answers
real questions, so the judge isn't grading its own generations) to
score faithfulness, relevancy, and correctness against the reference
answer.

This talks to real services (Ollama/Groq/Gemini for generation and
judging, plus the Chroma store built by `helper_scripts/ingest.py`), so it's
excluded from the default `pytest`/`make test` run (see pytest.ini)
and must be run explicitly:

    make eval

Was originally written against RAGAS; that implementation is kept
below, commented out, as a reference (see the bottom of this file).
"""

import json
import math
import os
import statistics
from datetime import datetime, timezone

import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from rag._09_common.config import load_config
from rag._07_evaluation.baseline import compare_to_baseline, format_comparison, load_baseline_mean_scores
from rag._05_generation.llm_client import LlmClient
from rag._05_generation.rag_pipeline import NO_MATCH_MESSAGE, RagPipeline
from rag._02_indexing.vector_store import VectorStore
from rag._01_ingestion.embeddings import EmbeddingGenerator
from rag._01_ingestion.glossary_extractor import load_glossary
from rag._04_retrieval.bm25_retriever import BM25Index
from rag._04_retrieval.dense_retriever import DenseRetriever
from rag._04_retrieval.reranker import Reranker
from rag._04_retrieval.retrieval_service import RetrievalService
from rag._03_routing.document_metadata import active_doc_topics, load_documents

pytestmark = pytest.mark.deepeval


def _extract_text(content) -> str:
    """Some chat models (e.g. Gemini 3.x) return .content as a list of
    content-block dicts instead of a plain string; DeepEval needs plain
    text, so flatten either shape into one."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


class _LangchainJudgeModel(DeepEvalBaseLLM):
    """Adapts one of this project's existing langchain chat clients
    (Ollama/Groq/Gemini) to DeepEval's judge-model interface, so
    DeepEval's metrics can drive the same provider choice
    config.evaluation.judge_provider already selects — no separate
    DeepEval-specific model integration needed."""

    def __init__(self, chat, model_name):
        self._chat = chat
        super().__init__(model=model_name)

    def load_model(self):
        return self._chat

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return _extract_text(self._chat.invoke(prompt).content)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        result = await self._chat.ainvoke(prompt)
        return _extract_text(result.content)

    def get_model_name(self) -> str:
        return self.name


# answer_correctness has no built-in DeepEval metric; GEval lets us
# define an equivalent custom LLM-judged criterion instead of dropping
# the metric.
def _build_metric(name, judge_model, threshold):
    if name == "faithfulness":
        return FaithfulnessMetric(model=judge_model, threshold=threshold, include_reason=True)
    if name == "answer_relevancy":
        return AnswerRelevancyMetric(model=judge_model, threshold=threshold, include_reason=True)
    if name == "context_precision":
        return ContextualPrecisionMetric(model=judge_model, threshold=threshold, include_reason=True)
    if name == "context_recall":
        return ContextualRecallMetric(model=judge_model, threshold=threshold, include_reason=True)
    if name == "answer_correctness":
        return GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output is factually correct and "
                "complete relative to the expected output."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=judge_model,
            threshold=threshold,
        )
    raise ValueError(f"Unknown evaluation metric: {name!r}")


@pytest.fixture(scope="module")
def config():
    return load_config()


class _CapturingRetrievalService:
    """Wraps a real RetrievalService and remembers the chunks its last
    call returned, so evaluation_rows can reuse the exact chunks
    pipeline.answer() retrieved internally instead of calling
    retrieve() a second time for the same question."""

    def __init__(self, retrieval_service):
        self._retrieval_service = retrieval_service
        self.last_chunks = []

    def retrieve(self, query, metadata_filter=None, top_k=None, min_reranker_score=None, category_override=None):
        self.last_chunks, matched_topic = self._retrieval_service.retrieve(
            query, metadata_filter, top_k=top_k, min_reranker_score=min_reranker_score,
            category_override=category_override,
        )
        return self.last_chunks, matched_topic


@pytest.fixture(scope="module")
def pipeline_and_retriever(config):
    embedding_generator = EmbeddingGenerator(
        config.embedding.model_name, batch_size=config.embedding.batch_size
    )
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    # doc_topics must be passed here the same way dashboard.py passes it
    # to the production RetrievalService — otherwise this evaluation
    # silently scores GLOBAL-mode-only behavior and never exercises the
    # PRIMARY path real users' topic-matched queries actually take.
    doc_topics = active_doc_topics(load_documents(config.documents.documents_file))
    retrieval_service = RetrievalService(
        dense_retriever=DenseRetriever(vector_store),
        embedding_generator=embedding_generator,
        top_k=config.retrieval.top_k,
        min_reranker_score=config.retrieval.min_reranker_score,
        min_relevance_score=config.retrieval.min_relevance_score,
        bm25_index=BM25Index.from_vector_store(vector_store) if config.retrieval.hybrid.enabled else None,
        hybrid_candidates=config.retrieval.hybrid.candidates,
        dense_weight=config.retrieval.hybrid.dense_weight,
        lexical_weight=config.retrieval.hybrid.lexical_weight,
        reranker=Reranker(config.retrieval.rerank.model)
        if config.retrieval.rerank.enabled
        else None,
        rerank_candidates=config.retrieval.rerank.candidates,
        doc_topics=doc_topics,
        primary_top_k=config.retrieval.primary_top_k,
        low_confidence_min_score=config.retrieval.low_confidence_min_score,
        topic_match_threshold=config.retrieval.topic_match_threshold,
    )
    capturing_retrieval_service = _CapturingRetrievalService(retrieval_service)
    llm_client = LlmClient.from_config(config.llm)
    glossary = load_glossary(config.documents.glossary_file)
    pipeline = RagPipeline(
        retrieval_service=capturing_retrieval_service, llm_client=llm_client, glossary=glossary
    )
    return pipeline, capturing_retrieval_service


@pytest.fixture(scope="module")
def judge_model(config):
    provider = config.evaluation.judge_provider
    if provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set; required for judge_provider: gemini")
        chat = ChatGoogleGenerativeAI(
            model=config.evaluation.judge_model,
            temperature=config.evaluation.judge_temperature,
        )
    elif provider == "groq":
        api_key_env = config.llm.groq.api_key_env
        if not os.environ.get(api_key_env):
            pytest.skip(f"{api_key_env} not set; required for judge_provider: groq")
        chat = ChatGroq(
            api_key=os.environ[api_key_env],
            model=config.evaluation.judge_model,
            temperature=config.evaluation.judge_temperature,
        )
    elif provider == "ollama":
        chat = ChatOllama(
            base_url=config.llm.ollama.base_url,
            model=config.evaluation.judge_model,
            temperature=config.evaluation.judge_temperature,
            # qwen3 is a reasoning model that emits an extended <think>
            # chain-of-thought by default; on CPU that alone pushes a
            # single judge call past even a 600s timeout for no benefit —
            # this is a structured extraction/scoring task, not one that
            # needs deep reasoning.
            reasoning=False,
        )
    else:
        raise ValueError(f"Unknown evaluation.judge_provider: {provider!r}")
    return _LangchainJudgeModel(chat, config.evaluation.judge_model)


def _load_test_cases(path):
    # strict=False: the fixture file has literal newlines inside some
    # multi-paragraph answer strings, which strict JSON parsing rejects
    # as invalid control characters.
    with open(path, "r", encoding="utf-8") as test_file:
        return json.load(test_file, strict=False)


@pytest.fixture(scope="module")
def evaluation_rows(config, pipeline_and_retriever):
    """The plain {question, answer, contexts} rows the pipeline
    produced, kept separate from the LLMTestCase list DeepEval
    consumes so the per-row results written by the test below can
    report the actual question text next to each metric score."""
    pipeline, capturing_retriever = pipeline_and_retriever
    test_cases = _load_test_cases(config.evaluation.test_data_path)

    rows = []
    total = len(test_cases)
    for case_index, case in enumerate(test_cases, start=1):
        question = case["question"]
        # This loop runs the real (possibly CPU-bound) generation
        # pipeline once per test case before DeepEval's own judging step
        # even starts, so it's the part most likely to look "stuck" —
        # print progress as each answer completes rather than only
        # after all of them are done.
        print(f"\n[{case_index}/{total}] answering: {question[:80]}", flush=True)
        result = pipeline.answer(question)
        print(f"[{case_index}/{total}] answered ({len(result.answer_text)} chars)", flush=True)
        chunks = capturing_retriever.last_chunks
        rows.append(
            {
                "user_input": question,
                "response": result.answer_text,
                "retrieved_contexts": [chunk.text for chunk in chunks] or [""],
                "reference": case["answer"],
                "category": case.get("category", "normal"),
            }
        )
    return rows


@pytest.fixture(scope="module")
def deepeval_test_cases(evaluation_rows):
    return [
        LLMTestCase(
            input=row["user_input"],
            actual_output=row["response"],
            expected_output=row["reference"],
            retrieval_context=row["retrieved_contexts"],
        )
        for row in evaluation_rows
    ]


def test_pipeline_answers_pass_judge_thresholds(
    config, evaluation_rows, deepeval_test_cases, judge_model
):
    metric_names = config.evaluation.metrics
    metrics = [_build_metric(name, judge_model, config.evaluation.min_score) for name in metric_names]

    result = deepeval_evaluate(
        test_cases=deepeval_test_cases,
        metrics=metrics,
        # Mirrors RAGAS's judge_max_workers: free-tier judge APIs are
        # rate-limited per minute, not CPU-bound, so concurrency is
        # capped rather than left at DeepEval's default of 20.
        async_config=AsyncConfig(run_async=True, max_concurrent=config.evaluation.judge_max_workers),
        # A single slow/rate-limited row shouldn't abort the whole
        # batch; a failed metric surfaces as MetricData.error instead,
        # which the NaN handling below still fails the run on.
        error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # Index test_results the same order deepeval_test_cases/evaluation_rows
    # were built in, so scores can be matched back to the right row.
    test_results_by_index = sorted(result.test_results, key=lambda test_result: test_result.index)
    total_questions = len(evaluation_rows)
    completed_questions = len(test_results_by_index)
    print(f"\nQuestions completed: {completed_questions}/{total_questions}")
    if completed_questions < total_questions:
        # deepeval_evaluate() still appends a test_result for every row
        # even when one of its metrics errors (that's how the NaN rows
        # above surface) — ignore_errors=True never drops a row's
        # result entirely. A row missing here means deepeval raised
        # out of the batch (TimeoutError/CancelledError) before
        # finishing it, which would already show up as a traceback
        # elsewhere in this same pytest run. Naming the missing
        # question(s) here points at which traceback to go read,
        # rather than guessing a cause we can't actually confirm.
        present_indices = {test_result.index for test_result in test_results_by_index}
        missing_indices = [i for i in range(total_questions) if i not in present_indices]
        print(
            f"  {len(missing_indices)} question(s) never returned a result — "
            "deepeval_evaluate() raised before finishing them (see the traceback "
            "elsewhere in this run for the actual cause, e.g. a timeout):"
        )
        for index in missing_indices:
            print(f"    [{index}] {evaluation_rows[index]['user_input'][:80]}")
    scores_by_row = [
        {
            metric_data.name: (
                float("nan") if metric_data.score is None or metric_data.error else metric_data.score
            )
            for metric_data in (test_result.metrics_data or [])
        }
        for test_result in test_results_by_index
    ]
    # DeepEval's display names ("Faithfulness", "Correctness [GEval]", ...)
    # differ from our config metric names ("faithfulness",
    # "answer_correctness", ...); map back using each metric's __name__.
    metric_name_by_display_name = {
        metric.__name__: name for metric, name in zip(metrics, metric_names)
    }
    scores_by_row = [
        {metric_name_by_display_name[display_name]: score for display_name, score in row_scores.items()}
        for row_scores in scores_by_row
    ]

    # "unanswerable" questions (no valid answer exists in the policy
    # docs) are expected to make the pipeline refuse — faithfulness
    # then judges the refusal sentence against empty/irrelevant context
    # and always scores it ~0, since a refusal is a meta-statement about
    # the context, not a factual claim grounded in it. That's a property
    # of the metric, not a pipeline defect, so folding these rows into
    # the mean would punish correct refusals as if they were
    # hallucinations. They're scored separately below instead.
    normal_indices = [
        index for index, row in enumerate(evaluation_rows) if row["category"] != "unanswerable"
    ]
    unanswerable_indices = [
        index for index, row in enumerate(evaluation_rows) if row["category"] == "unanswerable"
    ]

    # A per-row NaN (e.g. faithfulness on a row where the pipeline
    # refused, so there's no context to check claims against — see the
    # unanswerable-rows comment above, which applies just as much to a
    # normal/indirect-category row that happens to trigger a refusal)
    # is scored as 0 rather than excluded: for a normal-category
    # question, a refusal IS a failure to answer, so it should drag
    # the mean down the same as a bad answer would, not disappear from
    # the average as if it were never asked.
    mean_scores = {}
    nan_treated_as_zero_counts = {}
    for name in metric_names:
        scored_values = [scores_by_row[index][name] for index in normal_indices]
        nan_treated_as_zero_counts[name] = sum(1 for value in scored_values if math.isnan(value))
        mean_scores[name] = statistics.mean(0.0 if math.isnan(value) else value for value in scored_values)
    print(f"\nDeepEval mean scores (normal questions only, n={len(normal_indices)}): {mean_scores}")
    if any(nan_treated_as_zero_counts.values()):
        print(
            "NaN rows treated as 0 in mean per metric (judge couldn't score, e.g. a "
            f"refusal with no context to check): {nan_treated_as_zero_counts}"
        )

    # Per-question scores, so a failing/low mean can be traced back to
    # which specific answers the judge marked down and why — the mean
    # alone doesn't say whether every answer is mediocre or a few are
    # badly hallucinating while the rest are fine.
    rows = [
        {
            "question": row["user_input"],
            "answer": row["response"],
            "category": row["category"],
            **scores_by_row[row_index],
        }
        for row_index, row in enumerate(evaluation_rows)
    ]

    print("\nPer-question DeepEval scores:")
    for row in rows:
        scores_text = ", ".join(f"{name}={row[name]:.3f}" for name in metric_names)
        print(f"  [{row['category']}] {row['question'][:80]}\n    {scores_text}")

    unanswerable_refusals = [
        {
            "question": evaluation_rows[index]["user_input"],
            "answer": evaluation_rows[index]["response"],
            "correctly_refused": evaluation_rows[index]["response"] == NO_MATCH_MESSAGE,
        }
        for index in unanswerable_indices
    ]
    if unanswerable_refusals:
        correct_refusal_count = sum(row["correctly_refused"] for row in unanswerable_refusals)
        print(
            f"\nUnanswerable-question refusals: {correct_refusal_count}/"
            f"{len(unanswerable_refusals)} correctly refused"
        )
        for row in unanswerable_refusals:
            marker = "✅" if row["correctly_refused"] else "❌"
            print(f"  {marker} {row['question'][:80]}")
            if not row["correctly_refused"]:
                print(f"      answered instead: {row['answer'][:150]}")

    baseline_mean_scores = load_baseline_mean_scores(config.evaluation.baseline_path)
    comparison = compare_to_baseline(mean_scores, baseline_mean_scores)
    if comparison is None:
        print(
            f"\nNo baseline at {config.evaluation.baseline_path} — run "
            "`make save-eval-baseline` to compare future runs against this one."
        )
    else:
        print(f"\nVs baseline ({config.evaluation.baseline_path}):\n{format_comparison(comparison)}")

    with open(config.evaluation.results_path, "w", encoding="utf-8") as results_file:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "judge_provider": config.evaluation.judge_provider,
                "judge_model": config.evaluation.judge_model,
                "min_score": config.evaluation.min_score,
                "mean_scores": mean_scores,
                "nan_treated_as_zero_counts": nan_treated_as_zero_counts,
                "baseline_comparison": comparison,
                "unanswerable_refusals": unanswerable_refusals,
                "rows": rows,
            },
            results_file,
            indent=2,
        )

    # Gate on normal-question mean scores only — the unanswerable rows
    # are reported separately above (see unanswerable_refusals) since
    # a correct refusal always scores ~0 on these metrics, which would
    # otherwise look like a hallucination and drag the mean down for
    # the wrong reason. A NaN mean (every row in that metric failed to
    # score, e.g. a judge/rate-limit outage) fails the same as a score
    # of 0: both mean the metric produced no usable signal, so the run
    # must stop rather than get silently reported as PASSED.
    failing_metrics = {
        name: score
        for name, score in mean_scores.items()
        if math.isnan(score) or score < config.evaluation.min_score
    }
    assert not failing_metrics, (
        f"DeepEval mean score(s) below min_score={config.evaluation.min_score} for "
        f"normal questions (NaN means the judge failed to score every row for "
        f"that metric): {failing_metrics}. See {config.evaluation.results_path} "
        "for the full per-question breakdown."
    )


# ---------------------------------------------------------------------------
# Original RAGAS implementation, kept for reference. Superseded by the
# DeepEval version above; not imported or executed.
# ---------------------------------------------------------------------------
#
# import sys
# import types
#
# # ragas unconditionally imports langchain_community.chat_models.vertexai
# # at import time, but that submodule was dropped from the langchain-community
# # version this project uses elsewhere. Nothing here uses Vertex AI (the judge
# # and embeddings are both local, via Ollama/HuggingFace), so stub the missing
# # submodule out before ragas (or anything importing it) loads.
# if "langchain_community.chat_models.vertexai" not in sys.modules:
#     _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
#     _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
#     sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub
#
# from langchain_huggingface import HuggingFaceEmbeddings
# from ragas import EvaluationDataset, evaluate
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from ragas.llms import LangchainLLMWrapper
# from ragas.metrics import (
#     AnswerCorrectness,
#     AnswerRelevancy,
#     ContextPrecision,
#     ContextRecall,
#     Faithfulness,
# )
# from ragas.run_config import RunConfig
#
# _METRIC_CLASSES = {
#     "faithfulness": Faithfulness,
#     "answer_relevancy": AnswerRelevancy,
#     "answer_correctness": AnswerCorrectness,
#     "context_precision": ContextPrecision,
#     "context_recall": ContextRecall,
# }
#
#
# @pytest.fixture(scope="module")
# def judge_llm(config):
#     provider = config.evaluation.judge_provider
#     if provider == "gemini":
#         if not os.environ.get("GOOGLE_API_KEY"):
#             pytest.skip("GOOGLE_API_KEY not set; required for judge_provider: gemini")
#         chat = ChatGoogleGenerativeAI(
#             model=config.evaluation.judge_model,
#             temperature=config.evaluation.judge_temperature,
#         )
#     elif provider == "groq":
#         api_key_env = config.llm.groq.api_key_env
#         if not os.environ.get(api_key_env):
#             pytest.skip(f"{api_key_env} not set; required for judge_provider: groq")
#         chat = ChatGroq(
#             api_key=os.environ[api_key_env],
#             model=config.evaluation.judge_model,
#             temperature=config.evaluation.judge_temperature,
#         )
#     elif provider == "ollama":
#         chat = ChatOllama(
#             base_url=config.llm.ollama.base_url,
#             model=config.evaluation.judge_model,
#             temperature=config.evaluation.judge_temperature,
#             reasoning=False,
#         )
#     else:
#         raise ValueError(f"Unknown evaluation.judge_provider: {provider!r}")
#     return LangchainLLMWrapper(chat)
#
#
# @pytest.fixture(scope="module")
# def judge_embeddings(config):
#     return LangchainEmbeddingsWrapper(
#         HuggingFaceEmbeddings(model_name=config.embedding.model_name)
#     )
#
#
# @pytest.fixture(scope="module")
# def evaluation_dataset(evaluation_rows):
#     ragas_rows = [
#         {key: value for key, value in row.items() if key != "category"} for row in evaluation_rows
#     ]
#     return EvaluationDataset.from_list(ragas_rows)
#
#
# def test_pipeline_answers_pass_ragas_judge_thresholds_OLD(
#     config, evaluation_rows, evaluation_dataset, judge_llm, judge_embeddings
# ):
#     metrics = [_METRIC_CLASSES[name]() for name in config.evaluation.metrics]
#
#     result = evaluate(
#         dataset=evaluation_dataset,
#         metrics=metrics,
#         llm=judge_llm,
#         embeddings=judge_embeddings,
#         run_config=RunConfig(
#             timeout=config.evaluation.judge_timeout_seconds,
#             max_workers=config.evaluation.judge_max_workers,
#         ),
#         raise_exceptions=False,
#     )
#     # ... (scoring/reporting logic identical in shape to the DeepEval
#     # version above — see git history for the full original body)
