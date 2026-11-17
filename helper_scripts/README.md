# helper_scripts/

Standalone, run-from-the-CLI scripts that support the RAG app in
`rag/` but aren't part of its request-time path. Most are wrapped by a
`make` target (see the root `Makefile`) — run those instead of calling
a script directly unless you need its flags.

| File | Run via | What it does |
|---|---|---|
| `ingest.py` | `make ingest` | Builds/rebuilds the Chroma vector index from `knowledge_documents/`: load PDFs, chunk, embed, write to Chroma. Also usable for a one-off `main.py ingest`. |
| `download_models.py` | `make download-models` | Downloads/verifies the embedding and reranker models into the local HuggingFace cache. |
| `log_pipeline_answers.py` | — | Runs the real RAG pipeline (retrieval + LLM, no judge) over every question in the configured test set and writes a plain question/expected/actual log to `logs/pipeline_answers.json`. Useful for eyeballing answers without paying for a judge pass. |
| `run_experiment.py` | `make experiment` / `make experiments-all` | Runs one retrieval evaluation with optional config overrides and logs it as an MLflow run under `mlops/`. See `config/experiments.yaml` for named hyperparameter sets. |
| `filter_low_faithfulness.py` | `make test-eval-failures` | Filters the evaluation question set down to whatever scored faithfulness < 1 in the last judge run, so those can be re-checked without re-running the whole suite. |
| `save_baseline.py` | `make save-eval-baseline` | Snapshots the latest retrieval/judge eval results as the baseline future runs are compared against. |
| `llm_diagnostics/check_gemini_latency.py` | `make llm-test` | Times a single bare Gemini call, to isolate provider latency from RAGAS's own retry/timeout behavior. |
| `llm_diagnostics/check_groq_limits.py` | `make llm-test` | Makes a cheap Groq call and prints the rate-limit headers, to check quota headroom without waiting to get rate-limited. |

All scripts read configuration via `rag._09_common.config.load_config()`
(`config/config.yaml`), so behaviour changes there rather than via
script flags, except where a script exposes its own CLI flags (see
`run_experiment.py --help`).
