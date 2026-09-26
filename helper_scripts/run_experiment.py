"""Runs one retrieval evaluation and logs it as an MLflow run, so
different models/hyperparameters can be compared over time.

Unlike the retrieval half of `make eval` (a plain pytest pass/fail
gate, no MLflow involved), every run of this script becomes one comparable
MLflow run, tracked locally in ./mlops/mlflow.db (SQLite, no server) with
artifacts under ./mlops/mlruns/.

    python helper_scripts/run_experiment.py
    python helper_scripts/run_experiment.py --run-name baseline
    python helper_scripts/run_experiment.py --set retrieval.top_k=3 --set retrieval.hybrid.dense_weight=0.7

Or put a named set of hyperparameters in a YAML file (see
experiments.yaml for the format) and pick one by name instead of
typing --set flags:

    python helper_scripts/run_experiment.py --overrides-file config/experiments.yaml --experiment low_threshold

Only classic IR metrics (Recall/Precision/MRR/NDCG@k, via
rag/_07_evaluation/evaluate_retrieval.py) are logged here -- RAGAS is never
invoked by this script, so a run never costs a judge-model call and
never touches tests/test-results/judge_results.json.

An override that changes embedding.model_name or any chunking.* value
changes what the index itself must contain, not just how it's
searched -- config.chroma.persist_directory (the shared store the live
app reads) is never rebuilt or overwritten for this. Instead,
_ensure_matching_index() detects the mismatch and transparently builds
(once) or reuses (on later runs with the same settings) a dedicated,
fingerprint-cached index under ./chroma-db-store/chroma_store_experiments/, running
`helper_scripts/ingest.py`'s pipeline automatically first if that exact
embedding/chunking combination hasn't been indexed yet. Each run's
"index_source" tag in MLflow records which store it actually used.

View results with:

    mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db
    (or: make mlflow-ui)
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import yaml

from rag._09_common.config import load_config
from rag._07_evaluation.config_overrides import apply_overrides, config_to_dict
from rag._07_evaluation.evaluate_retrieval import evaluate_retrieval
from rag.pipeline_factory import build_retrieval_service
from helper_scripts.ingest import run_ingestion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLOPS_DIR = PROJECT_ROOT / "mlops"
# A single local SQLite file, not a directory tree — still fully
# local/no server (same as MLflow's older file-store backend, which
# newer MLflow versions warn is deprecated), just one file instead of
# a growing folder of run subdirectories.
MLFLOW_TRACKING_URI = f"sqlite:///{MLOPS_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = MLOPS_DIR / "mlruns"
MLFLOW_EXPERIMENT_NAME = "cmrit-rag-retrieval"

# Dedicated, fingerprint-cached Chroma indexes for experiment runs
# whose embedding/chunking overrides don't match config.yaml's own
# settings — see _ensure_matching_index(). Kept entirely separate from
# config.chroma.persist_directory (the shared store the live app
# reads) so an experiment run never mutates production data.
EXPERIMENT_INDEX_CACHE_ROOT = PROJECT_ROOT / "chroma-db-store" / "chroma_store_experiments"


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


def _index_signature(config) -> dict:
    """The subset of config that determines what a Chroma index
    actually contains (as opposed to retrieval.* hyperparameters,
    which only change how an already-built index is searched). Two
    configs with the same signature can safely share one index."""
    return {
        "embedding_model": config.embedding.model_name,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "separators": list(config.chunking.separators),
        "source_directory": config.documents.source_directory,
    }


def _fingerprint_signature(signature: dict) -> str:
    payload = json.dumps(signature, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _ensure_matching_index(config, base_signature: dict) -> tuple:
    """Makes sure config.chroma.persist_directory actually points at an
    index built with config's own embedding/chunking settings before
    retrieval runs against it.

    If config's signature matches base_signature (config.yaml's own,
    unoverridden settings — what config.chroma.persist_directory is
    assumed to already hold via the last `make ingest`), this is a
    no-op: the run reuses the shared production store as-is.

    Otherwise (an --set/--overrides-file override touched
    embedding.model_name or chunking.*), the shared store would silently
    misrepresent this run -- retrieval would run against vectors built
    by a *different* embedding model than the one just configured. To
    avoid that, this redirects config.chroma.persist_directory to a
    dedicated store under EXPERIMENT_INDEX_CACHE_ROOT, keyed by a
    fingerprint of the signature, and runs ingestion into it via
    helper_scripts/ingest.py's run_ingestion() -- but only the first time that
    exact signature is requested; a later run with the same
    embedding/chunking settings reuses the cached index instead of
    re-ingesting from scratch.

    Returns (index_source, rebuilt) for logging: index_source is
    "shared" or "dedicated:<fingerprint>", rebuilt is whether ingestion
    actually ran this call.
    """
    signature = _index_signature(config)
    if signature == base_signature:
        return "shared", False

    fingerprint = _fingerprint_signature(signature)
    persist_directory = str(EXPERIMENT_INDEX_CACHE_ROOT / fingerprint)
    config.chroma.persist_directory = persist_directory

    already_built = (Path(persist_directory) / "chroma.sqlite3").exists()
    if already_built:
        print(f"Reusing cached index for this run's embedding/chunking settings: {persist_directory}")
        return f"dedicated:{fingerprint}", False

    print(
        f"This run's embedding/chunking settings differ from config.yaml's -- "
        f"building a dedicated index at {persist_directory} (fingerprint {fingerprint}) ..."
    )
    run_ingestion(config)
    return f"dedicated:{fingerprint}", True


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
    # Untouched snapshot of config.yaml's own settings, taken before any
    # override is applied -- _ensure_matching_index compares against
    # this to decide whether config.chroma.persist_directory (the
    # shared production store) still matches what this run needs, or
    # whether a dedicated index has to be built/reused instead.
    base_signature = _index_signature(copy.deepcopy(config))

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

    index_source, index_rebuilt = _ensure_matching_index(config, base_signature)

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
                "index_source": index_source,
                "index_rebuilt_this_run": index_rebuilt,
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
