.PHONY: install download-embedding-model ingest run run-debug test test-ragas test-retrieval-eval metrics clean

install:
	pip install -r requirements.txt

# Downloads the sentence-transformers embedding model into the local
# cache (safe to re-run; it's a no-op if already downloaded).
download-embedding-model:
	python scripts/download_embedding_model.py

# Builds/rebuilds the Chroma vector index from policies_docs/.
# Run this once after install, and again whenever the PDFs change.
ingest:
	python scripts/ingest.py

run:
	streamlit run app.py  --server.port 8502

# Same as run, but with verbose Streamlit logging plus an in-app debug
# panel showing the query, retrieved chunks (source doc/page), and
# their similarity scores.
run-debug:
	RAG_DEBUG=true streamlit run app.py --logger.level=debug

test:
	pytest tests/

# Live RAGAS judge evaluation: scores real pipeline answers against
# tests/test-question-qnswers.json using the judge model configured
# in config.yaml. Needs Ollama running (both ollama.model and
# evaluation.judge_model pulled) and an ingested Chroma store (`make
# ingest`), so it's excluded from the default `make test`.
test-ragas:
	pytest tests/evaluation/test_ragas_judge.py -m ragas -v

# Retrieval-only evaluation: Recall/Precision/MRR/NDCG @ top_k for
# test cases with a "relevant_sources" ground truth, scored against
# the real retriever with no LLM call — much faster than
# `make test-ragas` and a good first check after tuning retrieval.
# Needs an ingested Chroma store (`make ingest`).
test-retrieval-eval:
	pytest tests/evaluation/test_retrieval_eval.py -m retrieval_eval -v

# Runs both evaluation suites back to back: retrieval-only metrics
# (Recall/Precision/MRR/NDCG, see test-retrieval-eval) and the RAGAS
# judge metrics (faithfulness/answer_relevancy/etc, see test-ragas).
# Needs an ingested Chroma store and, for test-ragas, Ollama running.
metrics: test-retrieval-eval test-ragas

# Removes generated caches/build artifacts (see .gitignore) — bytecode,
# test/lint caches, coverage reports, packaging output. Does NOT touch
# chroma_store/ or logs/: those are runtime data, not disposable cache,
# and chroma_store/ in particular is expensive to rebuild (`make ingest`).
clean:
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -not -path './.venv/*' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov *.egg-info build dist
