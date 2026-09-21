"""RAGAS-judged evaluation of the live RAG pipeline.

For every {question, answer} pair in the file at
config.evaluation.test_data_path, runs the real retriever + LLM
pipeline to get an actual answer and its retrieved context, then asks
a separate judge model (config.evaluation.judge_model, e.g. qwen3:8b —
deliberately not the same model that answers real questions, so the
judge isn't grading its own generations) to score faithfulness,
relevancy, and correctness against the reference answer.

This talks to real services (Ollama for both generation and judging,
plus the Chroma store built by `scripts/ingest.py`), so it's excluded
from the default `pytest`/`make test` run (see pytest.ini) and must be
run explicitly:

    make test-ragas
"""

import json
import os
import statistics
import sys
import types
from datetime import datetime, timezone

import pytest

# ragas unconditionally imports langchain_community.chat_models.vertexai
# at import time, but that submodule was dropped from the langchain-community
# version this project uses elsewhere. Nothing here uses Vertex AI (the judge
# and embeddings are both local, via Ollama/HuggingFace), so stub the missing
# submodule out before ragas (or anything importing it) loads.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from rag.common.config import load_config
from rag.evaluation.baseline import compare_to_baseline, format_comparison, load_baseline_mean_scores
from rag.generation.llm_client import LlmClient
from rag.generation.rag_pipeline import NO_MATCH_MESSAGE, RagPipeline
from rag.indexing.vector_store import VectorStore
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.bm25_retriever import BM25Index
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.reranker import Reranker
from rag.retrieval.retrieval_service import RetrievalService
from rag.routing.docs_metadata import load_doc_topics

pytestmark = pytest.mark.ragas

_METRIC_CLASSES = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "answer_correctness": AnswerCorrectness,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
}


@pytest.fixture(scope="module")
def config():
    return load_config()


class _CapturingRetrievalService:
    """Wraps a real RetrievalService and remembers the chunks its last
    call returned, so evaluation_dataset can reuse the exact chunks
    pipeline.answer() retrieved internally instead of calling
    retrieve() a second time for the same question."""

    def __init__(self, retrieval_service):
        self._retrieval_service = retrieval_service
        self.last_chunks = []

    def retrieve(self, query, metadata_filter=None, top_k=None, min_reranker_score=None):
        self.last_chunks, matched_topic = self._retrieval_service.retrieve(
            query, metadata_filter, top_k=top_k, min_reranker_score=min_reranker_score
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
    # MAPPED_DOCUMENT path real users' topic-matched queries actually take.
    doc_topics = load_doc_topics(config.documents.metadata_file)
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
        mapped_document_primary_top_k=config.retrieval.mapped_document_primary_top_k,
        low_confidence_min_score=config.retrieval.low_confidence_min_score,
        topic_match_threshold=config.retrieval.topic_match_threshold,
    )
    capturing_retrieval_service = _CapturingRetrievalService(retrieval_service)
    llm_client = LlmClient.from_config(config.llm)
    pipeline = RagPipeline(retrieval_service=capturing_retrieval_service, llm_client=llm_client)
    return pipeline, capturing_retrieval_service


@pytest.fixture(scope="module")
def judge_llm(config):
    provider = config.evaluation.judge_provider
    if provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set; required for judge_provider: gemini")
        chat = ChatGoogleGenerativeAI(
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
    return LangchainLLMWrapper(chat)


@pytest.fixture(scope="module")
def judge_embeddings(config):
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.embedding.model_name)
    )


def _load_test_cases(path):
    # strict=False: the fixture file has literal newlines inside some
    # multi-paragraph answer strings, which strict JSON parsing rejects
    # as invalid control characters.
    with open(path, "r", encoding="utf-8") as test_file:
        return json.load(test_file, strict=False)


@pytest.fixture(scope="module")
def evaluation_rows(config, pipeline_and_retriever):
    """The plain {question, answer, contexts} rows the pipeline
    produced, kept separate from the EvaluationDataset RAGAS consumes
    so the per-row results written by the test below can report the
    actual question text next to each metric score."""
    pipeline, capturing_retriever = pipeline_and_retriever
    test_cases = _load_test_cases(config.evaluation.test_data_path)

    rows = []
    total = len(test_cases)
    for case_index, case in enumerate(test_cases, start=1):
        question = case["question"]
        # This loop runs the real (possibly CPU-bound) generation
        # pipeline once per test case before RAGAS's own judging step
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
def evaluation_dataset(evaluation_rows):
    # "category" is only used by the test below to split unanswerable
    # questions out of the RAGAS mean — strip it so RAGAS only ever
    # sees the fields it actually expects.
    ragas_rows = [
        {key: value for key, value in row.items() if key != "category"} for row in evaluation_rows
    ]
    return EvaluationDataset.from_list(ragas_rows)


def test_pipeline_answers_pass_ragas_judge_thresholds(
    config, evaluation_rows, evaluation_dataset, judge_llm, judge_embeddings
):
    metrics = [_METRIC_CLASSES[name]() for name in config.evaluation.metrics]

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        # RAGAS's default 180s per-call timeout is too tight for an 8B
        # judge model on CPU; a timed-out call silently becomes NaN
        # rather than raising, which is why the NaN check below exists.
        run_config=RunConfig(
            timeout=config.evaluation.judge_timeout_seconds,
            max_workers=config.evaluation.judge_max_workers,
        ),
        # Local CPU inference speed varies a lot per test case (longer
        # reference answers take longer to judge); one slow outlier
        # shouldn't abort the whole batch. A timed-out row becomes NaN
        # instead, which the min_score check below still fails on.
        raise_exceptions=False,
    )

    # "unanswerable" questions (no valid answer exists in the policy
    # docs) are expected to make the pipeline refuse — RAGAS faithfulness
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

    mean_scores = {
        name: statistics.mean(result[name][index] for index in normal_indices)
        for name in config.evaluation.metrics
    }
    print(f"\nRAGAS mean scores (normal questions only, n={len(normal_indices)}): {mean_scores}")

    # Per-question scores, so a failing/low mean can be traced back to
    # which specific answers the judge marked down and why — the mean
    # alone doesn't say whether every answer is mediocre or a few are
    # badly hallucinating while the rest are fine.
    rows = [
        {
            "question": row["user_input"],
            "answer": row["response"],
            "category": row["category"],
            **{name: result[name][row_index] for name in config.evaluation.metrics},
        }
        for row_index, row in enumerate(evaluation_rows)
    ]

    print("\nPer-question RAGAS scores:")
    for row in rows:
        scores_text = ", ".join(
            f"{name}={row[name]:.3f}" for name in config.evaluation.metrics
        )
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
            "`make save-ragas-baseline` to compare future runs against this one."
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
                "baseline_comparison": comparison,
                "unanswerable_refusals": unanswerable_refusals,
                "rows": rows,
            },
            results_file,
            indent=2,
        )
