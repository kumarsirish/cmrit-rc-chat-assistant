# RAG Assistant

A config-driven Retrieval-Augmented Generation app: answers questions
grounded strictly in whatever documents (PDF, Word, plain text, Excel,
PowerPoint) are indexed from `knowledge_documents/`. It refuses to
answer questions the documents don't cover, links to the source
document/page for every answer, and includes an in-app Admin UI for
uploading/superseding documents and rebuilding the index without
touching the command line. This deployment happens to index CMRIT
Research Committee documents, but nothing in the code assumes that —
swap the document set and `config/config.yaml`'s `app`/`documents`
sections to deploy it for something else.

## How it works (data flow)

**Ingestion** (`make ingest`, run offline ahead of time, or via the
Admin UI's "Rebuild index" button):

```
knowledge_documents/*.{pdf,docx,txt,xlsx,pptx}
    -> document_loader.py   (extract text, one entry per page/sheet/slide/whole file)
    -> chunker.py            (recursive, URL-safe splitting into overlapping chunks)
    -> embeddings.py         (sentence-transformers -> one vector per chunk)
    -> vector_store.py       (write chunk text + vector + metadata into Chroma)
```

`helper_scripts/ingest.py` also tags every chunk with the topic name,
canonical link, category, and curated keywords its filename matches in
`config/documents.json` (see "Document metadata" below), and the BM25
lexical index (`rag/_04_retrieval/bm25_retriever.py`) is rebuilt
in-memory from the same Chroma collection at process startup — nothing
extra is persisted for it.

**Query time** (every question typed into the app):

```
user question
    -> retrieval_service.py (embed the question; topic-match it against
                              either one category's documents or the
                              whole corpus, depending on the Chat page's
                              category picker — see "Retrieval, in
                              detail" below. Dense (Chroma) + BM25
                              search in parallel, fused by Reciprocal
                              Rank Fusion, optionally reranked with a
                              cross-encoder, filtered by the relevance
                              threshold, truncated to top_k)
    -> rag_pipeline.py      (if nothing relevant survived: refuse, no
                              LLM call — and offer a rephrased question
                              instead. Otherwise: build a numbered
                              context block from the retrieved chunks
                              and send it + a strict system prompt to
                              whichever LLM provider config.yaml selects)
    -> llm_client.py        (provider-agnostic: builds a LangChain chat
                              model via init_chat_model() for
                              config.yaml's llm.provider — "ollama"
                              (local) or "groq" (hosted))
    -> response_processor.py (validate/normalize the model's "[N]"
                              citations, map each retrieved chunk to a
                              source reference)
    -> query_logger.py       (append one JSON line — question, every
                              candidate chunk + score, the LLM's raw
                              reply, timings — to this process's
                              logs/query_log_<timestamp>.jsonl)
    -> dashboard.py          (render the answer + cited sources, each
                              linking to/downloadable from its source
                              document)
```

Session/auth is a side channel that wraps the query above:

```
browser session -> user_identity.py (validate email) -> session_store.py
                    (MySQL: create/touch a session row, log every
                     message + its rating + timing, 5-minute
                     inactivity timeout)
```

## Architecture

Each RAG step is its own module under `rag/`, so it can be read,
tested, and swapped independently:

| Step | Module |
|---|---|
| 1. Document loading | `rag/_01_ingestion/document_loader.py` |
| 2. Chunking (recursive, URL-safe) | `rag/_01_ingestion/chunker.py` |
| 3. Sentence embeddings | `rag/_01_ingestion/embeddings.py` |
| — Acronym glossary extraction (one-off tool) | `rag/_01_ingestion/glossary_extractor.py` |
| 3. Vector storage (Chroma) | `rag/_02_indexing/vector_store.py` |
| 4a. Dense (embedding) retrieval | `rag/_04_retrieval/dense_retriever.py` |
| 4a. Lexical (BM25) retrieval | `rag/_04_retrieval/bm25_retriever.py` |
| 4a. Dense+BM25 fusion (RRF) | `rag/_04_retrieval/hybrid_retriever.py` |
| 4b. Cross-encoder reranking | `rag/_04_retrieval/reranker.py` |
| 4c. Document metadata (filename/category/topic/link) | `rag/_03_routing/document_metadata.py` |
| 4c. Topic detection (query -> document(s)) | `rag/_03_routing/topic_matcher.py` |
| 4c. Query glossary expansion | `rag/_03_routing/query_expander.py` |
| 4. Retrieval orchestration (FOCUSSED_CATEGORY/GLOBAL, PRIMARY+SECONDARY) | `rag/_04_retrieval/retrieval_service.py` |
| 5. Session & chat history (MySQL, 5 min timeout) | `rag/_08_auth/session_store.py` |
| — Email validation (identifies each user) | `rag/_08_auth/user_identity.py` |
| 6. LLM chat client (provider-agnostic: Ollama or Groq) | `rag/_05_generation/llm_client.py` |
| 6. Context block formatting | `rag/_05_generation/context_builder.py` |
| 6. Rephrase suggestion on refusal | `rag/_05_generation/rephrase_suggester.py` |
| 6. Citation validation / answer assembly | `rag/_06_response/response_processor.py` |
| 7. Grounded answer orchestration | `rag/_05_generation/rag_pipeline.py` |
| — Per-query structured JSON debug log | `rag/_09_common/query_logger.py` |
| Chat UI (all Streamlit chat-page code) | `rag/_10_ui/dashboard.py` |
| Admin UI (upload/supersede/rebuild/history) | `rag/_10_ui/admin_page.py` |
| Configuration | `config/config.yaml`, loaded by `rag/_09_common/config.py` |
| Retrieval/judge evaluation | `rag/_07_evaluation/` |

`rag/_02_indexing/` builds/stores the index; `rag/_04_retrieval/`
queries it at question time. `dense_retriever.py`, `bm25_retriever.py`,
`hybrid_retriever.py`, and `reranker.py` are generic retrieval
mechanisms with no document-specific knowledge; `retrieval_service.py`
is the only module that knows about topics, categories, or the
FOCUSSED_CATEGORY/PRIMARY/SECONDARY/GLOBAL policy — see its module
docstring for the full flow.

`helper_scripts/ingest.py` runs steps 1-3 (plus topic tagging) to
(re)build the vector index — called both by `make ingest` and by the
Admin UI's "Rebuild index" button (`rag/_10_ui/admin_page.py`, run on a
background thread so the UI stays responsive while it works).
`helper_scripts/download_models.py` pre-fetches the embedding and
reranker models. `rag/_10_ui/dashboard.py` and `admin_page.py` are the
only modules with Streamlit UI code; `main.py` at the project root is
the entry point Streamlit runs (`streamlit run main.py`), which calls
`dashboard.py`'s `main()`.

## Document metadata (`config/documents.json`)

One JSON record per document — this is the single source of truth for
everything that isn't the document's raw content: which category/folder
it lives in, when it was uploaded, whether it's still active or has
been superseded by a newer upload, and the topic/link/keyword fields
used to detect which document a query is about. Loaded/written by
`rag/_03_routing/document_metadata.py`.

```json
{
  "documents": [
    {
      "filename": "RC Norms.docx",
      "category": "Policies and norms",
      "upload_date": "2026-11-17T04:55:25.370199+00:00",
      "status": "active",
      "supersedes": null,
      "superseded_by": null,
      "link": "https://docs.google.com/...",
      "label": null,
      "keywords": null
    }
  ]
}
```

- `status` is `"active"` (currently indexed/routable) or `"archived"`
  (superseded — physically moved into that category's `_archived`
  subfolder, excluded from ingestion's directory scan and from topic
  matching). Only `rag._03_routing.document_metadata.active_doc_topics()`
  filters on this — every retrieval-facing consumer goes through it,
  so archived documents can never become routable again by accident.
- `label`, if set, overrides the display/match name that would
  otherwise be derived from the filename (extension stripped,
  underscores turned into spaces) — meant only for a filename that's a
  poor semantic-match target (an acronym, a version suffix, garbled
  export characters).
- `keywords` are curated terms (job titles, synonyms) a query might use
  that the document's own topic name doesn't contain — folded into both
  topic matching and the chunk's indexed text (see `search_text()` in
  `helper_scripts/ingest.py`).
- `helper_scripts/migrate_document_metadata.py` is a one-time migration
  script that produced this file from two now-retired predecessors
  (`docs-metadata.json` + `document_registry.json`) — kept for
  reference, not part of the normal ingest/admin flow.

## Retrieval, in detail

`rag/_04_retrieval/retrieval_service.py` is the most involved module, so
its own docstring is worth reading directly. There are two modes,
chosen per query from the Chat page's category picker alone — there is
no automatic category detection:

- **FOCUSSED_CATEGORY** — the user picked a specific category. The
  question is topic-matched only against that category's own documents
  (`EmbeddingTopicMatcher`, one instance built per category). Every
  topic clearing `retrieval.topic_match_threshold` counts as matched,
  not just the single best one — a query can legitimately be about more
  than one document in the same category.
- **GLOBAL** — the picker is set to "All categories" (or no category was
  given at all, e.g. from an evaluation script). Same topic-matching,
  scored against every active document in the whole corpus instead of
  one category's.

Within either mode, the same three-step flow runs, just scoped
differently:

1. **PRIMARY** — if the topic match maps to one or more documents,
   dense+BM25 search is restricted to the union of their chunks
   (`source_file` filter), up to `retrieval.primary_top_k` chunks kept.
   This pool is trusted by the topic match: relevance is judged by dense
   cosine score, not reranker score, and an optional reranker can
   reorder it but never evict a chunk outright.
2. **Fallback** — if nothing matched, or PRIMARY came back empty, a
   plain hybrid search runs over the same scope (one category, or the
   whole corpus) with no document restriction, judged by the full
   reranker-score threshold like a normal search.
3. **SECONDARY** — if PRIMARY found something but didn't fill
   `retrieval.top_k`, a second same-scope search (excluding PRIMARY's
   documents) fills the remaining slots.

Each of those three stages is itself: dense search + BM25 search, fused
by Reciprocal Rank Fusion (`RRF_K = 60`, from the original RRF paper),
optionally reranked by a cross-encoder (`retrieval.rerank`) over the
fused pool's best `rerank.candidates`, then filtered to
`min_reranker_score` (or `min_relevance_score`/dense-score for the
trusted PRIMARY pool). If nothing clears the threshold, the single
best-scoring candidate is still returned — flagged `low_confidence`,
shown to the user with a caveat — as long as it clears
`low_confidence_min_score`; otherwise the pipeline refuses outright.

The Chat page itself adds one more layer on top of `retrieve()`: if a
specific category comes back ungrounded, it's retried once more with
"all categories" before presenting a refusal — widening the search
rather than stopping at "not found in this one category."

Every query logs one summary line (`rag._04_retrieval.retrieval_service`
logger, INFO level) with the matched topic(s), mapped document(s), mode,
candidate/result counts, and final source documents. The fuller,
structured trace — every candidate considered and every reranker score,
not just a summary — goes to the per-query JSON log (see "Observability"
below), independent of this line.

Retrieval always runs on the question alone — conversation history
(see "Session & chat history" below) is passed to the answer LLM so it
can resolve references like "it"/"that" in its own reply, but it never
rewrites or otherwise affects the retrieval query.

## Observability

Two independent debug mechanisms, serving different purposes:

- **Per-query JSON log** (`logs/query_log_<launch-timestamp>.jsonl`,
  `rag/_09_common/query_logger.py`) — always on, one JSON object per
  line per question answered: the question, every stage's candidates
  (dense/lexical/fused scores), every reranker score, the chunks that
  survived, the LLM's raw reply, the final processed answer, and
  per-stage timings. A new file is started every app launch (the
  timestamp is fixed at process start), so restarting the app never
  appends to or overwrites a previous run's log. Wired into
  `RagPipeline` as an optional `log_query` callback (default: a no-op,
  so unit tests never hit disk); `main_pipeline.build_rag_pipeline()` is
  the one place that wires in the real file-writing function for the
  live app, `helper_scripts/run_experiment.py`, and
  `helper_scripts/log_pipeline_answers.py` alike.
- **`RAG_DEBUG=true`** (`make run-debug`) — shows an in-app "🐛 Debug:
  retrieval details" expander under each answer (query, every retrieved
  chunk's source/page/similarity score) and a "Debug mode: ON" sidebar
  caption. Independent of the JSON log above, which runs regardless of
  this flag.

## Setup

### 1. Create a virtual environment

Use a dedicated virtual environment rather than installing into the
system Python, so this project's pinned dependencies (see
`requirements.txt`) can't clash with anything else on the machine:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

`.venv/` is already excluded via `.gitignore`. Every command below
(`make install`, `make run`, `pytest`, ...) assumes this environment is
active — reactivate it (`source .venv/bin/activate`) in any new shell
before running them.

### 2. Install dependencies (including Chroma)

```bash
make install
```

This is just `pip install -r requirements.txt` — there is nothing extra
to install for Chroma itself. `chromadb` (via `langchain-chroma`) runs
**embedded, in-process**, persisting to the local
`chroma-db-store/chroma_store/` directory (see `chroma.persist_directory`
in `config/config.yaml`) — not a separate server you install or start.
The first time anything opens the collection (`make ingest`, or the app
itself), Chroma creates that directory automatically if it doesn't
exist yet.

### 3. Answer LLM (Ollama or Groq)

Which provider answers real user questions is `llm.provider` in
`config/config.yaml` — `"ollama"` (local) or `"groq"` (hosted). Only the
selected provider needs to actually be reachable at runtime; the other
provider's config block can stay in `config/config.yaml` unused.

**Ollama** (`llm.provider: "ollama"`) — make sure Ollama is running
locally with the configured model pulled:

```bash
ollama pull llama3.1:8b
```

**Groq** (`llm.provider: "groq"`) — no local server needed; export an
API key under the environment variable named by `llm.groq.api_key_env`
(default `GROQ_KEY`) before running the app or tests:

```bash
export GROQ_KEY="gsk_..."
```

### 4. Download/verify the local embedding and reranker models

```bash
make download-models
```

### 5. MySQL

MySQL must be reachable with the credentials in `config/config.yaml`
(or via the `RAG_DB_PASSWORD` environment variable). The database and
tables are created automatically the first time the app actually runs
(a plain HTTP health check does not trigger this — it needs a live
Streamlit session, i.e. someone opening the app in a browser).

The app only ever touches the single database named
`<app.name slugified>_sessions` (e.g. `rc_assistant_sessions` for the
default `app.name: "RC Assistant"`) — it never reads or writes any
other database on the server. The DB user in `config/config.yaml`
therefore only needs privileges scoped to that one database, not
server-wide access:

```sql
CREATE USER 'rag_app'@'%' IDENTIFIED BY 'your-password';
-- underscores escaped with `\_` so MySQL treats the name literally, not as a wildcard
GRANT ALL PRIVILEGES ON `rc\_assistant\_sessions`.* TO 'rag_app'@'%';
FLUSH PRIVILEGES;
```

If the account can't be granted `CREATE DATABASE` (e.g. a shared/managed
MySQL instance), create the database yourself first and grant
privileges on it instead of using the statement above.

Each user is asked for their email address before they can ask a
question (or a default is assumed per `app.default_user_email` if the
deployment skips that prompt) — it identifies their session and chat
history in MySQL.

### 6. Admin password

The Admin tab (document upload, rebuild index, clear history) is gated
by `admin.password` in `config/config.yaml`, overridable via the
`RAG_ADMIN_PASSWORD` environment variable so it doesn't have to live in
the committed file.

## Build the vector index

Run this once, and again whenever files in `knowledge_documents/`
change (or use the Admin UI's "Rebuild index now" button instead, which
runs the same pipeline on a background thread with a live progress
bar):

```bash
make ingest
```

## Run the app

```bash
make run
```

For verbose Streamlit logging plus the in-app retrieval debug panel
(see "Observability" above):

```bash
make run-debug
```

### Admin UI

The Admin tab (password-gated) has five sub-tabs:

- **Upload Documents** — upload a single file or a `.zip` of several
  into a chosen (or newly typed) category; optionally mark it as
  superseding an existing active document in that category, which
  archives the old one and physically moves its file aside.
- **Existing Documents** — every currently-indexed file grouped by
  category (linked to its canonical URL where one is on record), plus
  an "Archived (superseded)" list showing what replaced what and when.
  Also where "Rebuild index now" lives.
- **Misc Actions** — permanently clear chat history for every user.
- **System Info** — see `rag/_09_common/system_info.py`.
- **Recent Queries** — the most recently answered questions across
  every user's session, each with its star rating.

## Makefile targets

| Target | What it does |
|---|---|
| `make install` | Installs Python dependencies |
| `make download-models` | Downloads/verifies the local embedding and reranker models (safe to re-run) |
| `make ingest` | (Re)builds the Chroma vector index from `knowledge_documents/` |
| `make run` | Starts the Streamlit app |
| `make run-debug` | Starts the Streamlit app with `--logger.level=debug` and an in-app debug panel |
| `make test` | Runs the unit test suite (excludes the `eval` suite, which needs live services — see below) |
| `make eval` | Runs the retrieval-only IR metrics (Recall/Precision/MRR/NDCG@k, no LLM call) then the live DeepEval judge evaluation against the real pipeline (needs the judge provider reachable + an ingested Chroma store) |
| `make test-eval-failures` | Re-runs the judge only for questions that scored faithfulness < 1 in the last `make eval` run |
| `make save-eval-baseline` | Saves the latest `make eval` results (both retrieval and judge) as the baseline future runs are compared against |
| `make experiment` | Runs one retrieval evaluation (config.yaml as-is, no overrides) and logs it as an MLflow run — see "Experiment tracking" below |
| `make experiments-all` | Runs every named block in `config/experiments.yaml` back to back, then opens the MLflow UI automatically |
| `make mlflow-ui` | Opens the MLflow UI to compare experiment runs |
| `make llm-test` | Checks Gemini judge latency and Groq rate-limit headroom (standalone diagnostics, not pytest) |
| `make clean` | Removes caches/build artifacts (not `chroma-db-store/` or `logs/`) |
| `make clean-eval-results` | Removes `tests/test-results/*.json` (eval results + baselines) |

## Experiment tracking (MLflow)

The retrieval half of `make eval` is a pass/fail gate (did retrieval
stay above `min_recall`/`min_mrr`); it doesn't keep history. For
comparing *different* models/hyperparameters against each other over
time, `helper_scripts/run_experiment.py` runs the same classic IR
evaluation (Recall/Precision/MRR/NDCG@k) but logs the result as a
comparable MLflow run instead of just asserting a threshold. **The
judge model is never involved here** — no judge call, no live
generation pipeline call; only retrieval is scored, which keeps each
run fast.

Tracking is entirely local — no server, no Docker: run metadata goes in
`./mlops/mlflow.db` (SQLite), artifacts go under `./mlops/mlruns/`, and
any dedicated indexes built for non-default embedding/chunking settings
live under `./chroma-db-store/chroma_store_experiments/`. All three are
gitignored.

```bash
# Run with config.yaml exactly as it is:
make experiment

# Run every named block in config/experiments.yaml back to back, then open the MLflow UI:
make experiments-all

# Override one or more values for this run only, without editing config.yaml:
python helper_scripts/run_experiment.py --run-name my-change --set retrieval.top_k=3

# Or name a reusable hyperparameter set in config/experiments.yaml instead of typing --set flags:
python helper_scripts/run_experiment.py --overrides-file config/experiments.yaml --experiment low_threshold

# Compare runs:
make mlflow-ui
```

`--set` accepts any dotted path under `config.yaml`'s `retrieval`,
`embedding`, or `chunking` sections — repeat the flag for multiple
overrides. Overriding `embedding.model_name` or any `chunking.*` value
doesn't silently score against the wrong index: those settings change
what a Chroma index actually contains, so `run_experiment.py` builds (or
reuses a cached) dedicated index under
`./chroma-db-store/chroma_store_experiments/` rather than touching the
shared store the live app reads.

## Configuration

All tunables live in `config/config.yaml`, loaded by
`rag/_09_common/config.py` into typed dataclasses — no code changes
needed to retune the system.

| Section | Key | Meaning |
|---|---|---|
| `app` | `name` / `description` | Page title and intro text |
| `app` | `example_queries` | Fallback sidebar example questions when no evaluation data is configured |
| `app` | `no_match_message` | Exact refusal sentence shown (and told to the LLM to reply with verbatim) when nothing is found |
| `app` | `default_user_email` | Identity attributed to every session until real login exists |
| `admin` | `password` | Gates the Admin tab; override via `RAG_ADMIN_PASSWORD` |
| `llm` | `provider` | `"ollama"` or `"groq"` — which provider answers real questions |
| `llm.ollama` | `base_url` / `model` / `temperature` / `request_timeout_seconds` / `num_thread` / `num_predict` | Local Ollama connection and generation settings |
| `llm.groq` | `model` / `temperature` / `request_timeout_seconds` / `max_tokens` / `api_key_env` | Hosted Groq connection and generation settings |
| `embedding` | `model_name` / `batch_size` | Local sentence-transformers model for chunk/query embeddings, and ingestion batch size |
| `chroma` | `persist_directory` | On-disk location of the vector index (`collection_name` is derived from `app.name`, not set here) |
| `documents` | `source_directory` | Where source documents are read from, optionally arranged in category subfolders |
| `documents` | `documents_file` | `config/documents.json` — see "Document metadata" above |
| `documents` | `glossary_file` | Acronym -> full-phrase glossary used to expand jargon-free questions before retrieval |
| `documents` | `archived_subfolder_name` | Subfolder (under each category) a superseded document is moved into; ingestion skips it |
| `documents` | `ingestion_stats_file` | Records the last rebuild's duration, shown as an estimate before the next one |
| `chunking` | `chunk_size` / `chunk_overlap` / `separators` | Recursive splitter's target size, overlap, and split-priority order |
| `retrieval` | `top_k` | Final chunks sent to the LLM per question |
| `retrieval` | `min_reranker_score` | Minimum cross-encoder score to keep a chunk |
| `retrieval` | `low_confidence_min_score` | Floor for the single-best-candidate fallback when nothing clears `min_reranker_score` |
| `retrieval` | `min_relevance_score` | Dense-score threshold for PRIMARY's trusted pool (not reranker-gated) |
| `retrieval` | `primary_top_k` | Cap on PRIMARY chunks kept before remaining `top_k` slots go to SECONDARY |
| `retrieval` | `topic_match_threshold` | Minimum query-topic cosine similarity to count as a topic match |
| `retrieval.hybrid` | `enabled` / `candidates` / `dense_weight` / `lexical_weight` | Dense+BM25 fusion toggle, per-method candidate pool size, and RRF weighting |
| `retrieval.rerank` | `enabled` / `model` / `candidates` | Cross-encoder reranking toggle, model, and how many fused candidates it scores |
| `session` | `timeout_minutes` / `warning_seconds_before_timeout` / `max_history_messages` | Inactivity timeout, countdown warning lead time, and how many recent turns are sent to the LLM as context |
| `evaluation` | `judge_provider` / `judge_model` / `judge_temperature` / `judge_timeout_seconds` / `judge_max_workers` | External judge for `make eval`, kept separate from whichever `llm.provider` answers real questions |
| `evaluation` | `test_data_path` / `metrics` / `min_score` / `results_path` / `baseline_path` | Judge evaluation inputs/thresholds/outputs |
| `evaluation.retrieval` | `min_recall` / `min_mrr` / `results_path` / `baseline_path` | Retrieval-only IR evaluation thresholds/outputs |
| `database` | `host` / `port` / `user` / `password` | MySQL connection for session/chat history; password overridable via `RAG_DB_PASSWORD` |

## Tests

```bash
make test
```

`test_retrieval_eval.py` and `test_judge.py` need a live answer-LLM
provider and/or an ingested Chroma store and are excluded from
`make test` by default (`pytest.ini`); run them via `make eval`.
Unit tests for `rag_pipeline`/`retrieval_service` use fakes instead of
real network calls, and never write to the real `logs/` directory
(`RagPipeline`'s query-log hook defaults to a no-op unless explicitly
wired in).

The judge half of `make eval` runs `pytest -s` so its progress is
visible live instead of only appearing after the run finishes.
