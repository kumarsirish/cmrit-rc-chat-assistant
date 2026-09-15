"""Runs one retrieval evaluation and logs it as an MLflow run, so
different models/hyperparameters can be compared over time.

Unlike `make test-retrieval-eval` (a plain pytest pass/fail gate, no
MLflow involved), every run of this script becomes one comparable
MLflow run, tracked locally in ./mlflow.db (SQLite, no server) with
artifacts under ./mlruns/.

    python scripts/run_experiment.py
    python scripts/run_experiment.py --run-name baseline
    python scripts/run_experiment.py --set retrieval.top_k=3 --set retrieval.hybrid.dense_weight=0.7

Or put a named set of hyperparameters in a YAML file (see
experiments.yaml for the format) and pick one by name instead of
typing --set flags:

    python scripts/run_experiment.py --overrides-file experiments.yaml --experiment low_threshold

View results with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
    (or: make mlflow-ui)
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import yaml

from rag.common.config import load_config
from rag.evaluation.config_overrides import apply_overrides, config_to_dict
from rag.evaluation.evaluate_retrieval import evaluate_retrieval
from rag.indexing.vector_store import VectorStore
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.bm25_retriever import BM25Index
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.reranker import Reranker
from rag.retrieval.retrieval_service import RetrievalService
from rag.routing.docs_metadata import load_doc_topics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# A single local SQLite file, not a directory tree — still fully
# local/no server (same as MLflow's older file-store backend, which
# newer MLflow versions warn is deprecated), just one file instead of
# a growing folder of run subdirectories.
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "cmrit-rag-retrieval"

REPRODUCIBILITY_WARNING = (
    "chunking.chunk_size/chunk_overlap are logged from config.yaml but "
    "are not verified against what chroma_store/ was actually built "
    "with -- re-run `make ingest` after any chunking change, otherwise "
    "these params may not describe the index this run actually queried."
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run one MLflow-tracked retrieval evaluation.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override a config.yaml value for this run only, e.g. --set retrieval.top_k=3",
    )
    parser.add_argument(
        "--overrides-file",
        default=None,
        metavar="PATH",
        help="YAML file of named hyperparameter sets (see experiments.yaml) — use with --experiment",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        metavar="NAME",
        help="Which named block in --overrides-file to use",
    )
    parser.add_argument("--run-name", default=None, help="Label shown for this run in the MLflow UI (defaults to --experiment, if given)")
    return parser.parse_args()


def _load_file_overrides(path: str, experiment_name: str) -> list:
    """Reads --overrides-file and returns the named block's
    hyperparameters as "key.path=value" strings, same shape as --set,
    so both go through the same apply_overrides() call."""
    with open(path, "r", encoding="utf-8") as overrides_file:
        named_experiments = yaml.safe_load(overrides_file) or {}

    if experiment_name not in named_experiments:
        available = ", ".join(sorted(named_experiments)) or "(none defined)"
        raise SystemExit(f"No experiment {experiment_name!r} in {path!r}. Available: {available}")

    overrides = named_experiments[experiment_name] or {}
    return [f"{key}={value}" for key, value in overrides.items()]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=PROJECT_ROOT).strip()
    except Exception:
        return "unknown"


def _dataset_fingerprint(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def build_retrieval_service(config):
    vector_store = VectorStore(
        persist_directory=config.chroma.persist_directory,
        collection_name=config.chroma.collection_name,
    )
    embedding_generator = EmbeddingGenerator(config.embedding.model_name, batch_size=config.embedding.batch_size)
    bm25_index = BM25Index.from_vector_store(vector_store) if config.retrieval.hybrid.enabled else None
    doc_topics = load_doc_topics(config.documents.metadata_file)
    reranker = Reranker(config.retrieval.rerank.model) if config.retrieval.rerank.enabled else None
    retrieval_service = RetrievalService(
        dense_retriever=DenseRetriever(vector_store),
        embedding_generator=embedding_generator,
        top_k=config.retrieval.top_k,
        min_reranker_score=config.retrieval.min_reranker_score,
        min_relevance_score=config.retrieval.min_relevance_score,
        bm25_index=bm25_index,
        hybrid_candidates=config.retrieval.hybrid.candidates,
        dense_weight=config.retrieval.hybrid.dense_weight,
        lexical_weight=config.retrieval.hybrid.lexical_weight,
        reranker=reranker,
        rerank_candidates=config.retrieval.rerank.candidates,
        doc_topics=doc_topics,
        mapped_document_primary_top_k=config.retrieval.mapped_document_primary_top_k,
        topic_match_threshold=config.retrieval.topic_match_threshold,
    )
    return retrieval_service, vector_store


def _flatten_params(config_dict: dict, prefix: str = "") -> dict:
    """Turns the nested resolved-config dict into flat dotted
    key -> value pairs for MLflow's flat param store."""
    flat = {}
    for key, value in config_dict.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_params(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _prefixed_metrics(scores: dict, prefix: str) -> dict:
    return {f"{prefix}{name}": value for name, value in scores.items()}


def _ensure_experiment() -> None:
    # Runs' metrics/params/tags go in mlflow.db (sqlite); artifacts
    # (resolved config, per-question results) go on disk under
    # MLFLOW_ARTIFACT_ROOT — set explicitly on first creation, since a
    # sqlite tracking URI has no built-in default artifact location.
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=f"file:{MLFLOW_ARTIFACT_ROOT}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def main():
    args = _parse_args()
    config = load_config()

    file_overrides = []
    if args.overrides_file:
        if not args.experiment:
            raise SystemExit("--overrides-file requires --experiment NAME to pick which block to use")
        file_overrides = _load_file_overrides(args.overrides_file, args.experiment)

    # File overrides apply first, then --set — so a --set flag can
    # still tweak one value on top of a named experiment without
    # editing the file.
    apply_overrides(config, file_overrides + args.overrides)
    run_name = args.run_name or args.experiment

    with open(config.evaluation.test_data_path, "r", encoding="utf-8") as test_file:
        all_cases = json.load(test_file, strict=False)
    test_cases = [case for case in all_cases if "relevant_sources" in case]
    if not test_cases:
        raise SystemExit("No test cases with 'relevant_sources' to evaluate retrieval against.")

    retrieval_service, vector_store = build_retrieval_service(config)
    result = evaluate_retrieval(retrieval_service, test_cases, config.retrieval.top_k)

    print(f"Retrieval mean scores (k={result['k']}): {result['mean_scores']}")
    print(f"Retrieval mean scores by category: {result['mean_by_category']}")

    # database section holds MySQL credentials, not a retrieval
    # hyperparameter, and shouldn't be logged even locally.
    resolved_config = config_to_dict(config)
    resolved_config["database"]["password"] = "<redacted>"
    params = _flatten_params(resolved_config)
    params = {key: value for key, value in params.items() if not key.startswith("database.")}

    chroma_sqlite_path = Path(config.chroma.persist_directory) / "chroma.sqlite3"
    chroma_store_mtime = chroma_sqlite_path.stat().st_mtime if chroma_sqlite_path.exists() else None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _ensure_experiment()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        mlflow.log_metrics(_prefixed_metrics(result["mean_scores"], ""))
        for category, scores in result["mean_by_category"].items():
            mlflow.log_metrics(_prefixed_metrics(scores, f"{category}_"))

        mlflow.set_tags(
            {
                "git_commit": _git("rev-parse", "HEAD"),
                "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "dataset_hash": _dataset_fingerprint(config.evaluation.test_data_path),
                "dataset_num_cases": len(test_cases),
                "num_indexed_chunks": vector_store.count(),
                "chroma_store_mtime": chroma_store_mtime,
                "reproducibility_warning": REPRODUCIBILITY_WARNING,
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            resolved_config_path = Path(tmp_dir) / "resolved_config.json"
            resolved_config_path.write_text(json.dumps(resolved_config, indent=2))
            mlflow.log_artifact(str(resolved_config_path))

            results_path = Path(tmp_dir) / "retrieval_results.json"
            results_path.write_text(json.dumps(result, indent=2))
            mlflow.log_artifact(str(results_path))

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run logged: {run_id} (experiment: {MLFLOW_EXPERIMENT_NAME!r})")
        print(f"View with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        print(f"          (this is also what `make mlflow-ui` runs)")


if __name__ == "__main__":
    main()
