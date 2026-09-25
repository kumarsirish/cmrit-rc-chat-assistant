.PHONY: help install download-models ingest run run-debug test test-ragas-failures eval save-eval-baseline experiment experiments-all mlflow-ui llm-test clean

.DEFAULT_GOAL := help

# Self-documenting: lists every target below along with the ## comment
# on its own target line. Add a new target? Add a `## description`
# after the colon and it shows up here automatically.
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies from requirements.txt
	pip install -r requirements.txt

# Downloads the sentence-transformers embedding model and the
# reranker model into the local cache (safe to re-run; it's a no-op
# for whichever is already downloaded).
download-models: ## Download the embedding and reranker models into the local cache
	python scripts/download_models.py

# Builds/rebuilds the Chroma vector index from policies_docs/.
# Run this once after install, and again whenever the PDFs change.
ingest: ## Build/rebuild the Chroma vector index from policies_docs/
	python scripts/ingest.py

run: ## Start the Streamlit app
	streamlit run app.py  --server.port 8502

# Same as run, but with verbose Streamlit logging plus an in-app debug
# panel showing the query, retrieved chunks (source doc/page), and
# their similarity scores.
run-debug: ## Start the Streamlit app with debug logging and in-app debug panel
	RAG_DEBUG=true streamlit run app.py --logger.level=debug

test: ## Run the default test suite (excludes ragas/retrieval_eval)
	pytest tests/

# Re-runs the DeepEval judge only for questions that scored
# faithfulness < 1 (or NaN, e.g. a timed-out judge call) in the last
# tests/judge_results.json — a fast way to check, during debugging,
# whether a retrieval/prompt change actually fixed the answers that
# were broken, without paying for the whole suite again. Requires
# tests/judge_results.json to already exist (`make eval`).
# RAG_EVAL_RESULTS_PATH points this run's output at a scratch file so
# it doesn't overwrite the full-suite results/baseline comparison.
test-ragas-failures: ## Re-run the judge only for questions with faithfulness < 1 in the last run
	python scripts/filter_low_faithfulness.py
	@if [ -f tests/evaluation-question-answers.low-faithfulness.json ]; then \
		RAG_EVAL_TEST_DATA_PATH=tests/evaluation-question-answers.low-faithfulness.json \
		RAG_EVAL_RESULTS_PATH=tests/judge_results.low-faithfulness.json \
			pytest tests/evaluation/test_judge.py -m ragas -v -s; \
	fi

# Runs both evaluation suites back to back:
# 1. Retrieval-only IR metrics (Recall/Precision/MRR/NDCG @ top_k),
#    scored against the real retriever with no LLM call — needs an
#    ingested Chroma store (`make ingest`).
# 2. The live DeepEval judge evaluation: scores real pipeline answers
#    against tests/evaluation-question-answers.json using the judge
#    model configured in config.yaml — needs the judge provider
#    reachable (Ollama running with both models pulled, or a Groq/
#    Gemini API key in the environment) and an ingested Chroma store.
# -s disables pytest's output capture so the per-question progress
# prints in test_judge.py show up live instead of only appearing
# after the run finishes (or gets silently buffered on failure).
# Both are excluded from the default `make test`.
eval: ## Run retrieval IR metrics + the live DeepEval judge evaluation
	pytest tests/evaluation/test_retrieval_eval.py -m retrieval_eval -v
	pytest tests/evaluation/test_judge.py -m ragas -v -s

# Snapshots the latest tests/retrieval_results.json and
# tests/judge_results.json as the baselines that every later `make
# eval` run reports its delta against (see rag/evaluation/baseline.py).
# Re-run after a deliberate change you want to become the new
# reference point.
save-eval-baseline: ## Save the latest eval results as the baseline for future comparisons
	python scripts/save_baseline.py retrieval
	python scripts/save_baseline.py ragas

# Runs one retrieval evaluation and logs it as an MLflow run (params,
# metrics, resolved config) to ./mlflow.db, for comparing model/
# hyperparameter changes over time. Unlike the retrieval half of
# `make eval` this is not a pass/fail gate — it's the tool for
# deliberate experiments. For --set overrides or --run-name, call the
# script directly instead:
#   python scripts/run_experiment.py --set retrieval.top_k=3 --run-name my-experiment
experiment: ## Run one retrieval experiment and log it to MLflow
	python scripts/run_experiment.py

# Runs every named block in experiments.yaml as its own MLflow run, back
# to back, instead of picking one via --experiment. Block names are read
# from the file itself (not hardcoded here), so adding a new block to
# experiments.yaml picks it up automatically. A block whose
# embedding.model_name/chunking.* differs from config.yaml's own
# settings (see scripts/run_experiment.py's _ensure_matching_index)
# triggers a one-time dedicated re-ingest the first time it runs; every
# other block reuses the shared ./chroma_store and is fast. Opens the
# MLflow UI (foreground, Ctrl+C to stop) once every run has finished, so
# results are ready to compare without a separate `make mlflow-ui` call.
experiments-all: ## Run every named experiment in experiments.yaml, then open the MLflow UI
	@for exp in $$(python3 -c "import yaml; print(' '.join(yaml.safe_load(open('experiments.yaml'))))"); do \
		echo "=== Running experiment: $$exp ==="; \
		python scripts/run_experiment.py --overrides-file experiments.yaml --experiment "$$exp" || exit 1; \
	done
	$(MAKE) mlflow-ui

# Opens the MLflow UI over the local tracking store, to compare
# experiment runs logged by `make experiment`.
mlflow-ui: ## Open the MLflow UI over ./mlflow.db
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Standalone connectivity/rate-limit checks for the hosted LLM providers
# (Gemini judge latency, Groq rate-limit headers) — not pytest tests,
# just quick diagnostics to run before a slow make eval/experiment
# to confirm the provider is actually reachable and not quota-exhausted.
llm-test: ## Check Gemini judge latency and Groq rate-limit headroom
	python llm-diagnostics/check_gemini_latency.py
	python llm-diagnostics/check_groq_limits.py

# Removes generated caches/build artifacts (see .gitignore) — bytecode,
# test/lint caches, coverage reports, packaging output. Does NOT touch
# chroma_store/ or logs/: those are runtime data, not disposable cache,
# and chroma_store/ in particular is expensive to rebuild (`make ingest`).
clean: ## Remove caches/build artifacts (not chroma_store/ or logs/)
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -not -path './.venv/*' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov *.egg-info build dist
