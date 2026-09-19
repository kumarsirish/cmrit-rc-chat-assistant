# CMRIT College Policy Assistant (RAG)

A Retrieval-Augmented Generation app that answers questions about
CMRIT college policies, grounded strictly in the PDF documents in
`policies_docs/`. It refuses to answer questions the documents don't
cover, and links to the source PDF/page for every answer.

## How it works (data flow)

**Ingestion** (`make ingest`, run offline, ahead of time):

```
policies_docs/*.pdf
    -> document_loader.py   (extract text, one entry per PDF page)
    -> chunker.py            (recursive, URL-safe splitting into overlapping chunks)
    -> embeddings.py         (sentence-transformers -> one vector per chunk)
    -> vector_store.py       (write chunk text + vector + metadata into Chroma)
```

`scripts/ingest.py` also tags every chunk with the topic/URL it maps
to in `docs-metadata.json` (see "Topic matching" below), and a
`rag/retrieval/bm25_retriever.py` lexical index is rebuilt in-memory from the same
Chroma collection at query-server startup (nothing extra is persisted
for it).

**Query time** (every question typed into the app):

```
user question
    -> retriever.py     (embed the question; if it matches a topic in
                          docs-metadata.json that has a locally-indexed document,
                          search ONLY that document (MAPPED_DOCUMENT mode) —
                          otherwise search the whole collection (GLOBAL mode).
                          Either way: dense (Chroma) search + BM25 (lexical)
                          search in parallel, fused by Reciprocal Rank Fusion,
                          optionally reranked with a cross-encoder, filtered by
                          the relevance threshold, truncated to top-k. If a
                          MAPPED_DOCUMENT search comes up empty, fall back to
                          GLOBAL rather than returning nothing)
    -> rag_pipeline.py  (if nothing relevant survived: refuse, no LLM call.
                          otherwise: build a numbered context block from the
                          retrieved chunks and send it + a strict system prompt
                          to whichever LLM provider config.yaml selects)
    -> llm_client.py    (provider-agnostic: builds a LangChain chat model via
                          init_chat_model() for config.yaml's llm.provider —
                          "ollama" (local) or "groq" (hosted) — so the rest of
                          the app never touches a provider-specific client)
    -> response_processor.py (validate/normalize the model's "[N]" citations,
                          map each retrieved chunk to a source reference)
    -> rag/dashboard.py (render the answer + cited sources — chunks from the
                          same source document are merged into one citation
                          spanning their combined page range — each linking to
                          its source PDF/page)
```

Session/auth is a side channel that wraps the query above:

```
browser session -> user_identity.py (validate email) -> session_store.py
                    (MySQL: create/touch a session row, log every message,
                     5-minute inactivity timeout)
```

## Architecture

Each RAG step is its own module under `rag/`, so it can be read,
tested, and swapped independently:

| Step | Module |
|---|---|
| 1. Document loading (load PDFs) | `rag/ingestion/document_loader.py` |
| 2. Chunking (recursive, URL-safe) | `rag/ingestion/chunker.py` |
| 3. Sentence embeddings | `rag/ingestion/embeddings.py` |
| 3. Vector storage (Chroma) | `rag/indexing/vector_store.py` |
| 4a. Dense (embedding) retrieval | `rag/retrieval/dense_retriever.py` |
| 4a. Lexical (BM25) retrieval | `rag/retrieval/bm25_retriever.py` |
| 4a. Dense+BM25 fusion (RRF) | `rag/retrieval/hybrid_retriever.py` |
| 4b. Cross-encoder reranking | `rag/retrieval/reranker.py` |
| 4c. Topic detection (query -> docs-metadata.json entry) | `rag/routing/docs_metadata.py`, `rag/routing/topic_matcher.py` |
| 4. Retrieval orchestration (topic-mapped-document / GLOBAL policy, PRIMARY+SECONDARY allocation) | `rag/retrieval/retrieval_service.py` |
| 5. Session & chat history (MySQL, 5 min timeout) | `rag/auth/session_store.py` |
| — Email validation (identifies each user) | `rag/auth/user_identity.py` |
| 6. LLM chat client (provider-agnostic: Ollama or Groq, via LangChain `init_chat_model`) | `rag/generation/llm_client.py` |
| 6. Context block formatting | `rag/generation/context_builder.py` |
| 6. Citation validation / answer assembly | `rag/response/response_processor.py` |
| 7. Grounded answer orchestration | `rag/generation/rag_pipeline.py` |
| Frontend dashboard (all Streamlit UI code) | `rag/ui/dashboard.py` |
| Configuration | `config.yaml`, loaded by `rag/common/config.py` |

`rag/indexing/` builds/stores the index (the vector store); `rag/retrieval/`
queries it at question time. `rag/retrieval/dense_retriever.py`,
`bm25_retriever.py`, `hybrid_retriever.py`, and `reranker.py` are
generic retrieval mechanisms with no CMRIT-specific knowledge;
`retrieval_service.py` is the only module that knows about topics,
mapped documents, or the MAPPED_DOCUMENT/PRIMARY/SECONDARY/GLOBAL
policy — see its module docstring for the full flow.

`scripts/ingest.py` runs steps 1-3 (plus topic tagging) to (re)build
the vector index. `scripts/download_embedding_model.py` pre-fetches
the embedding model. `rag/dashboard.py` is the only module with
Streamlit UI/dashboard code; it wires the backend modules above
together and holds nothing that counts as RAG logic itself. `app.py`
at the project root is just the thin entry point Streamlit runs
(`streamlit run app.py`), which imports and calls `rag/dashboard.py`'s
`main()`.

### Retrieval, in detail

`rag/retriever.py` is the most involved module, so its own docstring
is worth reading directly. There are two modes, chosen fresh for every
query — no caller ever picks one explicitly:

- **MAPPED_DOCUMENT mode** — the question matches a topic in
  `docs-metadata.json` that has a locally-indexed document (embedding
  cosine-similarity match against each topic's name, see
  `EmbeddingTopicMatcher` in `topic_matcher.py`). That topic -> document
  mapping is trusted metadata, so retrieval is restricted to chunks
  from that one document only — both the dense and BM25 searches run
  with a `source_file` filter for it. Because every candidate that
  ever enters fusion/reranking already belongs to the mapped document,
  nothing outside it can end up in the result and nothing (fusion or
  reranking included) can replace it with a different document. If
  that document has nothing above the relevance threshold (the topic
  match was wrong, or the document just doesn't answer the question),
  retrieval falls back to GLOBAL mode rather than returning nothing.
- **GLOBAL mode** (fallback) — no topic matched, the matched topic has
  no local document (only an external link), or MAPPED_DOCUMENT mode
  came up empty. Searches the whole collection instead of one document.

Within either mode, the same steps run — only the search scope
(one document vs. the whole collection) and candidate pool size
differ:

1. **Dense search** — the question is embedded and compared against
   chunk vectors in Chroma (cosine similarity). Good at
   paraphrase/synonym matches, weak on exact acronyms/proper nouns/codes.
2. **BM25 (lexical) search** — the question's significant keywords
   (`bm25_retriever.py`, stopwords stripped) are matched against
   chunk term frequency. Good at exact keyword/acronym matches, blind
   to paraphrase. Kept even in MAPPED_DOCUMENT mode: the same
   acronym/proper-noun gap dense search has across the whole
   collection is just as real *within* one document (e.g. "TYL" is a
   strong BM25 hit but a weak dense one even restricted to its own
   source document).
3. **Reciprocal Rank Fusion (RRF)** — combines the two ranked lists by
   *rank*, not raw score, since cosine similarity and BM25 scores
   aren't on comparable scales. `RRF_K = 60` (the constant from the
   original RRF paper) damps how much a #1-vs-#2 rank difference
   matters, so it doesn't need per-corpus tuning. In MAPPED_DOCUMENT
   mode this only ever ranks chunks within the one mapped document.
4. **Cross-encoder reranking** (optional, `rerank.enabled`) — the
   fused pool is over-fetched to `rerank.candidates` chunks, and a
   `BAAI/bge-reranker-base` cross-encoder rescores each `(question,
   chunk)` pair jointly (much more accurate than comparing two
   independent embeddings, too slow to run over the whole collection).
   In MAPPED_DOCUMENT mode the pool it rescores only ever contains
   chunks from the mapped document, so it can reorder within it but
   never pull in, or be replaced by, a different document.
5. **Relevance threshold** — anything scoring below `min_reranker_score`
   (the cross-encoder's reranker_score, the authoritative relevance
   signal once reranking is on — see "Relevance filtering" below) is
   dropped before the LLM ever sees it. If nothing survives (and, for
   MAPPED_DOCUMENT mode, the GLOBAL fallback also finds nothing),
   `rag_pipeline.py` refuses without calling the LLM at all.

Every query logs one summary line (`rag.retrieval.retrieval_service` logger, INFO level)
with the detected topic, mapped document (if any), which mode ran,
candidate/result counts, and the final source documents — enough to
diagnose a retrieval failure without needing `RAG_DEBUG`'s fuller
per-stage trace.

There used to be a third mechanism — an additive "topic boost" applied
*after* global search instead of restricting search up front. It was
removed: since reranking fully rescored the boosted pool, the boost
only influenced which chunks reached the reranker, not its final
verdict, so a strong topic match could still lose to an unrelated
document. Restricting the search itself removes that failure mode
entirely rather than trying to out-tune it.

## Design decisions

| Decision | Why |
|---|---|
| A `docs-metadata.json` topic can map to more than one local PDF | `DocTopic.docs_filename` (`rag/routing/docs_metadata.py`) is always a list — `"docs"` in the JSON accepts a single filename string (kept as-is for every existing entry) or an array. MAPPED_DOCUMENT mode's scoped search then filters on `{"source_file": {"$in": [...]}}` across both the dense (Chroma) and BM25 backends, so a topic legitimately spread across a few PDFs (e.g. several RCPIE documents) can be searched as one trusted unit instead of forcing an artificial single-file split. |
| Curated `keywords` per topic (`docs-metadata.json`), used two ways | (1) `topic_matcher.py` embeds each keyword phrase alongside the topic name and takes the best match, so a query phrased nothing like the topic's formal title (e.g. "when do unused casual leaves expire" vs. "Leave Information, Application and Approval Process") can still clear `topic_match_threshold`. (2) `scripts/ingest.py`'s `search_text()` appends the same keywords onto a chunk's own text before embedding/BM25-indexing it (never onto the text shown to the LLM or cited), so even a GLOBAL-mode fallback search benefits from terms the document's own prose never uses. |
| One module per RAG step | Each stage (load, chunk, embed, store, retrieve, generate) can be read, tested, and replaced on its own without touching the others. |
| Hybrid retrieval (dense + BM25 fused by RRF), not dense-only | Dense embeddings miss exact acronyms/policy codes/proper nouns the model has too little signal to place correctly (e.g. "TYL"); BM25 alone misses paraphrases dense search catches naturally. Fusing by rank rather than picking a winner or normalizing scores lets whichever signal is actually strong for a given query carry the result, without the app guessing in advance which one that is. |
| Cross-encoder reranking as an optional second pass | A cross-encoder scoring `(query, chunk)` jointly is a strictly stronger relevance signal than comparing two independently-computed embeddings, but too slow to run over the whole collection — so it's applied only to the fused pool's best `rerank.candidates`, not the full corpus. |
| Topic detection from `docs-metadata.json`, embedding-based (`topic_matcher.py`) | The college already maintains a topic -> canonical-URL/filename mapping; matching a query against each topic's name by embedding cosine similarity (instead of literal keyword overlap) restricts retrieval to the right document even when the query paraphrases the topic name, and lets the app surface a source link even for topics with no locally-ingested PDF. Keyword overlap was tried first and dropped: it misses any topic phrased with different words than the topic's own name (e.g. "who do I email if I'm sick" sharing zero words with "Leave Information, Application and Approval Process"). The similarity threshold is configurable (`retrieval.topic_match_threshold`). |
| A topic match restricts search to the mapped document (MAPPED_DOCUMENT mode), not an additive score boost | An earlier design boosted a topic-matched chunk's score *after* global search, but reranking then fully rescored the pool and could still bury a correct topic match under an unrelated document — the boost only affected who got a chance in front of the reranker, not who won. Restricting the search itself to the mapped document removes cross-document competition entirely: nothing (fusion, threshold, or reranking) can substitute a different document once one is trusted. |
| MAPPED_DOCUMENT mode falls back to GLOBAL search if the mapped document has nothing above the relevance threshold | Topic detection is a similarity heuristic and can still pick the wrong topic. Trusting a wrong mapping with no escape hatch would turn a bad topic match into a hard refusal; falling back to a normal global search only when the trusted document comes up genuinely empty keeps the common case strict without making a heuristic's mistake fatal. |
| Answer LLM is provider-agnostic (`llm.provider: "ollama"` or `"groq"`), built via LangChain's `init_chat_model()` | `llm_client.py` never imports a provider-specific chat class — `init_chat_model(model, model_provider=...)` dispatches to `ChatOllama` or `ChatGroq` based on `config.yaml`, so switching providers (or adding another LangChain-supported one) is a config change, not a code change. Ollama runs fully offline/on-prem (no policy data leaves the machine); Groq is a hosted, much faster option for the same role, authenticated via an API key in the environment (`llm.groq.api_key_env`, default-configured as `GROQ_KEY`). Both providers' models are config values, not hard-coded. |
| Local `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings, not an Ollama embedding model | Keeps embedding independent of whatever LLM is configured, runs in-process (no extra network hop), and is small/fast enough to embed every chunk and every query on CPU. |
| Recursive, separator-cascade chunker (`chunker.py`) instead of a fixed character splitter | Tries paragraph breaks first, then line breaks, then sentence stops, and only hard-cuts by character count as a last resort — so a chunk boundary lands on a natural break in the text instead of mid-sentence whenever possible. |
| URLs are masked before sentence-stop splitting | Policy PDFs contain reference URLs (e.g. login portals). Splitting naively on `". "` would cut a URL in half at any `.` in the domain/path. The chunker temporarily replaces each URL with a placeholder, splits, then restores the URL — so links always survive intact in a single chunk. |
| `temperature: 0` | Deterministic, repeatable answers for the same question — appropriate for a policy lookup tool where consistency matters more than creative phrasing. |
| Strict system prompt + inline `[N]` citations (`rag_pipeline.py`, `response_processor.py`) | Second layer of grounding on top of the relevance-threshold refusal: the LLM is told to answer only from the supplied numbered context, cite every claim by source number, and say so explicitly if the context doesn't answer the question. `response_processor.py` then normalizes/strips any citation the model invents or writes in prose instead of `[N]`, so a citation left in the final answer is always real. |
| Conversation history passed to the LLM but never to retrieval | Retrieval always runs on the current question alone, so it stays deterministic and easy to evaluate independent of session state; history is only given to the LLM so it can resolve references like "it"/"that leave type" in its reply. |
| `llm.ollama.num_thread: 8`, set explicitly | On this machine (Ollama on CPU inside WSL2), Ollama's thread auto-detection undershot the available 8 cores. Pinning `num_thread` to the real core count (`nproc`) noticeably cut generation latency. Passed through per-request via `ChatOllama`'s `num_thread`; adjust to match whatever machine runs this. |
| Chroma, local/persistent (not an in-memory or hosted vector DB) | Matches the "local chroma db" requirement; index survives app restarts; no external service dependency. |
| BM25 index rebuilt in-memory at startup, not persisted separately | The corpus is small (a few hundred chunks), so `BM25Okapi` over every chunk in the Chroma collection builds fast enough to redo on every process start rather than maintaining a second on-disk index that could drift from Chroma. |
| MySQL for sessions instead of only in-memory Streamlit state | Streamlit's `session_state` is per-browser-tab and lost on refresh/restart; MySQL gives durable session/chat history across restarts and a shared store that scales to the expected <50 concurrent users. |
| Sessions identified by email, not anonymous UUID only | Ties conversation history in MySQL back to a real person (`rag/user_identity.py` validates the address) rather than being fully anonymous. |
| 5-minute inactivity timeout | Chosen per requirement; implemented as "no query in the last 5 minutes" (`sessions.last_active_at`), not a hard session lifetime — an actively-chatting user is never cut off. |
| Two separate evaluation suites, not one | `test_retrieval_eval.py` scores the retriever alone (classic IR metrics — Recall/Precision/MRR/NDCG@k) with no LLM call, so retrieval regressions are cheap and fast to catch. `test_ragas_judge.py` scores the full pipeline's generated answers with an external judge model (faithfulness/relevancy/correctness), which conflates retrieval and generation quality but is what actually matters to a user. Keeping them separate makes it possible to tell "retrieval is fine, generation is the problem" apart from the reverse. |
| RAGAS judge is Gemini (`gemini-2.5-flash`), not the same Ollama model that answers | A judge grading its own generations is a biased signal (it tends to rate its own phrasing/reasoning favorably). Gemini also runs faster/more capably than the local CPU-bound Ollama model would for a judge role. |
| No lambdas; plain functions/classes everywhere | Readability requirement — every transformation has a name and can be stepped through or unit-tested directly. |
| Configuration centralized in `config.yaml` | Every tunable (model names, chunk size/overlap, retrieval/rerank/hybrid settings, evaluation thresholds, timeouts, DB connection) lives in one file, loaded by `rag/config.py` into typed dataclasses — no magic numbers scattered through the code. |
| Citations from the same source document are merged, not just deduplicated | Chunking can split one document into overlapping/adjacent pieces (e.g. p. 40 and p. 40-41), which otherwise showed up as two separate numbered citations for the same file. `dashboard.py`'s `render_sources()` merges all cited chunks that share a `source_file` into one citation, using the first citation number and a page range spanning every merged chunk. |

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

This is just `pip install -r requirements.txt` — there is nothing
extra to install for Chroma itself. `chromadb` (pinned to `1.5.9`,
via `langchain-chroma`) is a regular Python package that runs
**embedded, in-process**, persisting to the local `chroma_store/`
directory (see `chroma.persist_directory` in `config.yaml`) — not a
separate server you install or start. The first time anything opens
the collection (`make ingest`, or the app itself), Chroma creates
`chroma_store/` on disk automatically if it doesn't exist yet.

### 3. Answer LLM (Ollama or Groq)

Which provider answers real user questions is `llm.provider` in
`config.yaml` — `"ollama"` (local, default) or `"groq"` (hosted).
Only the selected provider needs to actually be reachable at runtime;
the other provider's config block can stay in `config.yaml` unused.

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

### 4. Download/verify the local embedding model

```bash
make download-embedding-model
```

### 5. MySQL

MySQL must be reachable with the credentials in `config.yaml` (or via
the `RAG_DB_PASSWORD` environment variable). The database and tables
are created automatically the first time the app actually runs (a
plain HTTP health check does not trigger this — it needs a live
Streamlit session, i.e. someone opening the app in a browser).

Each user is asked for their email address before they can ask a
question; it identifies their session and chat history in MySQL
(`sessions.user_email`).

## Build the vector index

Run this once, and again whenever files in `policies_docs/` change:

```bash
make ingest
```

## Run the app

```bash
make run
```

For verbose Streamlit logging while troubleshooting (also enables an
in-app debug panel showing, per query, the detected topic, dense/BM25
candidate lists, fusion/rerank scores, and the final chunks):

```bash
make run-debug
```

## Makefile targets

| Target | What it does |
|---|---|
| `make install` | Installs Python dependencies |
| `make download-embedding-model` | Downloads/verifies the local `sentence-transformers` embedding model (safe to re-run) |
| `make ingest` | (Re)builds the Chroma vector index from `policies_docs/` |
| `make run` | Starts the Streamlit app |
| `make run-debug` | Starts the Streamlit app with `--logger.level=debug` and an in-app debug panel |
| `make test` | Runs the unit test suite (excludes `test-retrieval-eval`/`test-ragas`, which need live services — see below) |
| `make test-retrieval-eval` | Runs the retrieval-only IR metrics (Recall/Precision/MRR/NDCG@k) against the real Chroma store, no LLM call |
| `make test-ragas` | Runs the RAGAS judge evaluation against the real pipeline (needs Ollama + an ingested Chroma store) |
| `make metrics` | Runs `test-retrieval-eval` then `test-ragas` back to back |
| `make experiment` | Runs one retrieval evaluation (config.yaml as-is, no overrides) and logs it as an MLflow run — see "Experiment tracking" below |
| `make experiments-all` | Runs every named block in `experiments.yaml` back to back, then opens the MLflow UI automatically |
| `make mlflow-ui` | Opens the MLflow UI to compare experiment runs |

## Experiment tracking (MLflow)

`make test-retrieval-eval` is a pass/fail gate (did retrieval stay
above `min_recall`/`min_mrr`); it doesn't keep history. For comparing
*different* models/hyperparameters against each other over time —
"is `top_k=8` better than `top_k=5`?", "does swapping the reranker
model help?" — `scripts/run_experiment.py` runs the same classic IR
evaluation (Recall/Precision/MRR/NDCG@k) but logs the result as a
comparable MLflow run instead of just asserting a threshold.
**RAGAS is never involved here** — no judge-model call, no live
generation pipeline call, no `tests/ragas_results.json` write; only
retrieval is scored, which is what keeps each run fast.

Tracking is entirely local — no server, no Docker: run metadata
(params/metrics/tags) goes in `./mlflow.db` (SQLite), artifacts
(resolved config, full per-question results) go under `./mlruns/`, and
any dedicated indexes built for non-default embedding/chunking
settings (see below) live under `./chroma_store_experiments/`. All
three are gitignored; nothing here is shared or synced anywhere.

### Running an experiment

```bash
# Run with config.yaml exactly as it is:
make experiment

# Run every named block in experiments.yaml back to back, then open the MLflow UI automatically:
make experiments-all

# Override one or more values for this run only, without editing config.yaml:
python scripts/run_experiment.py --run-name my-change --set retrieval.top_k=3

# Or name a reusable hyperparameter set in experiments.yaml instead of typing --set flags:
python scripts/run_experiment.py --overrides-file experiments.yaml --experiment low_threshold
```

`--set` accepts any dotted path under `config.yaml`'s `retrieval`,
`embedding`, or `chunking` sections (e.g. `retrieval.hybrid.dense_weight`,
`retrieval.rerank.model`, `embedding.model_name`) — repeat the flag for
multiple overrides; each value is parsed to match that field's
existing type (int/float/bool/str) automatically. `experiments.yaml`
holds the same overrides as named, reusable blocks — see the file for
the format and the current presets (`baseline`, `low_threshold`,
`wide_top_k`, `lexical_heavy`, `alt_reranker`, `bge_base_embedding`).
`make experiments-all` reads the block names straight from that file
(nothing hardcoded in the `Makefile`), so a new block is picked up
automatically.

**Overriding `embedding.model_name` or any `chunking.*` value no
longer silently scores against the wrong index.** Those settings
change what a Chroma index actually contains, not just how it's
searched, so `scripts/run_experiment.py`'s `_ensure_matching_index()`
compares the run's settings against `config.yaml`'s own (what
`./chroma_store` — the shared store the live app reads — is assumed to
already hold via the last `make ingest`). If they match, the run
reuses that shared store untouched. If they differ (e.g.
`bge_base_embedding`'s `BAAI/bge-base-en-v1.5`), it transparently
builds — the *first* time that exact combination is requested only —
a dedicated index under `./chroma_store_experiments/<fingerprint>/`
via `scripts/ingest.py`'s own pipeline, and reuses that cached index on
every later run with the same settings. The production
`./chroma_store` is never rebuilt or overwritten by an experiment run.
Each run's `index_source` tag (see below) records which store it
actually used.

### Comparing runs

```bash
make mlflow-ui
```

Opens the MLflow UI (`http://localhost:5000`) over `./mlflow.db`
(`make experiments-all` also opens this automatically once every run
finishes). Each run shows:

- **Params** — the full resolved `config.yaml` (after any overrides),
  flattened to dotted keys, e.g. `retrieval.top_k`, `retrieval.rerank.model`
- **Metrics** — `recall`, `precision`, `reciprocal_rank` (MRR), `ndcg`,
  both overall and per-category (e.g. `normal_recall`, `unanswerable_precision`)
- **Tags** — `git_commit`, `git_branch`, `git_dirty` (uncommitted
  changes at run time), `dataset_hash`/`dataset_num_cases` (which
  version of `evaluation-question-answers.json` was scored),
  `num_indexed_chunks`, `chroma_store_mtime`, `index_source`
  (`"shared"` or `"dedicated:<fingerprint>"` — which store this run
  actually queried), and `index_rebuilt_this_run` (whether ingestion
  ran as part of this call)
- **Artifacts** — `resolved_config.json` (the full nested config, not
  just the flattened params) and `retrieval_results.json` (per-question
  scores, not just the means)

Sort/filter by any metric or tag in the UI to compare runs — e.g. sort
by `recall` descending to find the best-performing configuration tried
so far. Better IR metrics don't automatically mean better final
answers (more/different chunks retrieved is still only an input to
generation) — `make test-ragas` is the only place generation quality
(faithfulness/answer correctness) actually gets checked, so confirm
there before promoting an experiment's settings into `config.yaml`.

## Configuration

All tunables live in `config.yaml`, loaded by `rag/common/config.py`
into typed dataclasses — no code changes needed to retune the system.
`top_k`, the reranker relevance threshold, and the LLM temperature can
also be adjusted per-session from sliders in the app's sidebar, for
trying out different values without editing `config.yaml`; the sliders
start at whatever `config.yaml` sets and only affect that browser
session.

| Section | Key | Current value | Meaning |
|---|---|---|---|
| `llm` | `provider` | `ollama` | Which provider answers real user questions — `"ollama"` or `"groq"`. `rag_pipeline.py` and everything above it only ever call `LlmClient.generate_answer()`, so this is the only line that needs to change to switch providers |
| `llm.ollama` | `base_url` | `http://localhost:11434` | Ollama server address |
| `llm.ollama` | `model` | `llama3.1:8b` | Generation model; swap to any pulled Ollama model |
| `llm.ollama` | `temperature` | `0` | Sampling temperature — `0` for deterministic answers |
| `llm.ollama` | `request_timeout_seconds` | `300` | HTTP timeout for a generation call (CPU inference can be slow) |
| `llm.ollama` | `num_thread` | `8` | CPU threads Ollama uses per generation call; set to your machine's core count (`nproc`) |
| `llm.ollama` | `num_predict` | `220` | Hard cap on generated tokens per answer, so worst-case CPU-inference latency stays bounded |
| `llm.groq` | `model` | `qwen/qwen3.8-27b` | Generation model on Groq's hosted API |
| `llm.groq` | `temperature` | `0` | Sampling temperature — `0` for deterministic answers |
| `llm.groq` | `request_timeout_seconds` | `300` | HTTP timeout for a generation call |
| `llm.groq` | `max_tokens` | `800` | Answer-length cap (Groq is hosted, so this isn't a latency safety valve like `ollama.num_predict`, just a length cap) |
| `llm.groq` | `api_key_env` | `GROQ_KEY` | Name of the environment variable holding the Groq API key |
| `embedding` | `model_name` | `all-MiniLM-L6-v2` | Local sentence-transformers model for chunk/query embeddings |
| `embedding` | `batch_size` | `64` | Chunks embedded per batch during ingestion |
| `chroma` | `persist_directory` | `./chroma_store` | On-disk location of the vector index |
| `chroma` | `collection_name` | `college_policies` | Chroma collection name |
| `documents` | `source_directory` | `./policies_docs` | Where source PDFs are read from |
| `documents` | `metadata_file` | `./docs-metadata.json` | Topic -> canonical URL/filename lookup, used to restrict retrieval to a topic's mapped document (MAPPED_DOCUMENT mode) and for source links |
| `chunking` | `chunk_size` | `800` (characters) | Target chunk size — roughly one dense policy paragraph: big enough to keep a rule and its conditions together, small enough that several retrieved chunks still fit comfortably in the LLM's context window |
| `chunking` | `chunk_overlap` | `120` (characters, ~15%) | Overlap carried between consecutive chunks, so a sentence split across a boundary isn't lost to either side |
| `chunking` | `separators` | `["\n\n", "\n", ". ", " "]` | Split priority order (paragraph -> line -> sentence -> word) |
| `retrieval` | `top_k` | `8` | Final chunks returned per question, in both MAPPED_DOCUMENT and GLOBAL mode, after fusion/rerank/filtering. Raised from 5: the `wide_top_k` MLflow experiment (see "Experiment tracking" below) scored best across recall/precision/MRR/NDCG of everything compared, with no regression on the unanswerable-question category |
| `retrieval` | `min_reranker_score` | `0.35` | Minimum cross-encoder reranker score to keep a chunk (see "Relevance filtering" below for why this replaced the old dense-cosine `min_relevance_score`). If nothing survives, the pipeline refuses instead of guessing |
| `retrieval` | `mapped_document_primary_top_k` | `3` | MAPPED_DOCUMENT mode only: cap on how many of the mapped document's own ("PRIMARY") chunks are kept out of `top_k`, before any remaining slots go to a SECONDARY, cross-document search (see `retrieval_service.py`'s `MAPPED_DOCUMENT_PRIMARY_TOP_K`) |
| `retrieval` | `topic_match_threshold` | `0.5` | Minimum embedding cosine similarity between a query and a topic's name (`docs-metadata.json`) for `EmbeddingTopicMatcher` to count it as a match; below this, retrieval falls back to GLOBAL mode |
| `retrieval.hybrid` | `enabled` | `true` | GLOBAL mode only: turns on BM25 lexical search fused with dense search via RRF; disabled means dense-only |
| `retrieval.hybrid` | `candidates` | `10` | GLOBAL mode: chunks fetched from *each* of dense/lexical search before fusion. MAPPED_DOCUMENT mode uses a separate, smaller constant (`MAPPED_DOCUMENT_CANDIDATES = 6` in `retrieval_service.py`, not config — the search space is already one document) |
| `retrieval.hybrid` | `dense_weight` / `lexical_weight` | `0.5` / `0.5` | How much each search method's rank counts toward the fused RRF score. Equal by default |
| `retrieval.rerank` | `enabled` | `true` | Turns on the cross-encoder reranking pass over the candidate pool (fused GLOBAL pool, or the small MAPPED_DOCUMENT pool) |
| `retrieval.rerank` | `model` | `BAAI/bge-reranker-base` | Cross-encoder model (alt: `ms-marco-MiniLM-L-6-v2`) |
| `retrieval.rerank` | `candidates` | `8` | Top N of the candidate pool handed to the reranker; the final `top_k` is chosen from these |
| `session` | `timeout_minutes` | `5` | Inactivity timeout before a session is treated as expired |
| `session` | `max_history_messages` | `1` | Most recent chat turns sent to the LLM as conversation context |
| `evaluation` | `judge_provider` / `judge_model` | `gemini` / `gemini-2.5-flash` | External judge for `test-ragas`, kept separate from whichever `llm.provider` answers real questions so it isn't grading its own generations |
| `evaluation` | `test_data_path` | `./tests/evaluation-question-answers.json` | Question/reference-answer (+ optional `relevant_sources`) test cases used by both eval suites |
| `evaluation` | `min_score` | `0.5` | Minimum mean RAGAS metric score to pass `test-ragas` |
| `evaluation.retrieval` | `min_recall` / `min_mrr` | `0.6` / `0.5` | Minimum mean Recall@k / MRR to pass `test-retrieval-eval` |
| `database` | `host` / `port` / `user` / `database` | `127.0.0.1` / `3306` / `sirkumar` / `rag_sessions` | MySQL connection for session/chat history |
| `database` | `password` | — | Can be overridden with the `RAG_DB_PASSWORD` env var instead of editing the file |

## Tests

```bash
make test
```

Tests that need a live MySQL server are skipped automatically if one
is not reachable. `test_retrieval_eval.py` and `test_ragas_judge.py`
need a live answer-LLM provider (whichever `llm.provider` selects)
and/or an ingested Chroma store and are excluded from `make test` by
default (`pytest.ini`); run them via `make test-retrieval-eval` /
`make test-ragas` / `make metrics`. `rag_pipeline` and `retriever`
unit tests use fakes instead of real network calls.

`make test-ragas` runs `pytest -s` so its progress is visible live
instead of only appearing after the run finishes: a `[i/N] answering:
...` / `[i/N] answered (... chars)` line per test case while the real
pipeline generates each answer, then RAGAS's own progress bar while
the judge model scores them.

## Current metrics — what they say, and what's still weak

`make metrics` writes fresh numbers to `tests/retrieval_results.json`
(classic IR) and `tests/ragas_results.json` (LLM-judged), both also
readable from the app's System Info page.

**The numbers in this section predate `retrieval.top_k` moving from 5
to 8** (see "Configuration" above and the `wide_top_k` MLflow
experiment) — re-run `make metrics` to refresh them against the
current config.

### Retrieval-only (`test-retrieval-eval`, k=5, 24 labeled questions)

Retrieval was reworked to use a LangChain-backed dense retriever, a
cross-encoder reranker (`HuggingFaceCrossEncoder`) as the sole
relevance judge, and a recalibrated `min_reranker_score` (see
"Retrieval, in detail" above and `retrieval_service.py`). Current
numbers, run against the real Chroma store and the current 24-question
`evaluation-question-answers.json`:

| Category | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| `normal` (22 questions) | 0.977 | 0.445 | 0.701 | 0.728 |
| `unanswerable` (2 questions) | 1.000 | 1.000 | 1.000 | 1.000 |
| **Overall mean** | **0.979** | **0.491** | **0.726** | 0.750 |

**This passes the configured minimums** (`min_recall: 0.6`,
`min_mrr: 0.5`) comfortably. Precision is the weakest number: `top_k`
is 5, but most questions have only one or two truly relevant chunks,
so several correctly-ranked-but-lower-relevance chunks fill out the
remaining slots — this is a `top_k` sizing effect, not a sign
retrieval is picking wrong documents (recall/MRR are high).

### RAGAS judge (`test-ragas`, gemini-2.5-flash judge, current 4-question set)

These numbers are from before the MAPPED_DOCUMENT/GLOBAL retrieval
rework above and haven't been rerun yet (`make test-ragas` takes
~10-30 minutes against 4 questions); treat them as the prior
baseline, not current fact, until it's rerun.

```
faithfulness:       0.93   (min_score: 0.5 — pass)
answer_relevancy:   0.65   (min_score: 0.5 — pass)
answer_correctness: 0.45   (min_score: 0.5 — FAIL)
context_precision:  0.37   (min_score: 0.5 — FAIL)
context_recall:     0.50   (min_score: 0.5 — pass, right on the floor)
```

`make test-ragas` currently **fails** on this test set (it passed
before only because the committed result predated the test set growing
from 1 to 4 questions). The pattern is consistent with the retrieval
gaps above: **faithfulness stays high** — the model reliably sticks to
whatever context it's given and doesn't fabricate — but
**context_precision (0.37) and answer_correctness (0.45) are the worst
scores**, meaning the context handed to the LLM is frequently only
partially relevant, and the model can't produce a fully correct answer
from partially-wrong context no matter how faithful it is to that
context. This directly corroborates the retrieval-only findings: the
`multiple_related_documents` category and the reranker dropping
correct-but-acronym-heavy chunks are very likely the same root cause
showing up here as low context_precision.

### What needs to be improved

1. **The ground truth for TYL questions assumes multi-document
   recall, which MAPPED_DOCUMENT mode deliberately doesn't provide** —
   `evaluation-question-answers.json` lists `HoD Handbook - CMRIT.pdf` as a
   "relevant source" for TYL questions because it mentions TYL in
   passing; the authoritative document per `docs-metadata.json` is
   `TYL - Faculty Orientation...pdf` alone. Now that retrieval treats
   the topic mapping as authoritative and single-document by design,
   either the ground truth should be updated to reflect that
   philosophy (relevant_sources = the mapped document only, for
   single-topic questions), or `min_recall` needs to account for
   mapped-document questions being scored differently than open
   multi-document ones. This is a definitional decision, not a
   retrieval bug — flagging it rather than changing the test data
   unilaterally.
2. **The GLOBAL fallback (for a bad topic match, or no match at all)
   still has the pre-existing weaknesses hybrid+rerank always had** —
   no mechanism to diversify across source documents for a question
   that genuinely spans more than one *unrelated* policy doc (the
   `multiple_related_documents` category). This is now partially
   addressed for the case where several PDFs are really one topic
   (`DocTopic.docs_filename` accepts a list, and MAPPED_DOCUMENT mode's
   scoped search filters on all of them via `{"source_file": {"$in":
   [...]}}` — see "Design decisions" above), but that only helps when
   the topic mapping in `docs-metadata.json` is set up that way ahead
   of time. An ad-hoc question spanning two independently-topic-mapped
   documents still falls through to plain GLOBAL search with no
   diversification.
3. **Test set now has 24 labeled cases** (`normal` and `unanswerable`),
   up from the original 4 — the `multiple_related_documents` category
   from earlier runs is no longer present in the current file, so
   item 2 above (GLOBAL fallback diversification) is currently
   unvalidated by `test-retrieval-eval`; re-adding cases for it would
   close that gap.
4. **RAGAS's Faithfulness metric can't verify a claim that names its
   source document, even when the claim is correct** — reproduced
   directly against the live judge (Gemini 2.5 Flash) while debugging a
   low-scoring leave-policy question: `retrieved_contexts` handed to
   RAGAS is plain chunk text with no source metadata attached, so a
   sentence like "According to the Faculty Handbook, unutilized
   vacation leaves lapse at the end of the vacation period" gets marked
   unfaithful — not because the fact is wrong, but because the judge's
   NLI check can't find "Faculty Handbook" anywhere in the unlabeled
   context it's checking against. This is a harness limitation, not a
   pipeline defect; fixing it would mean prefixing each context string
   passed to RAGAS with its source document name (e.g. `[Source:
   Faculty Handbook-CMRIT.docx.pdf]`) so attribution claims become
   verifiable.
5. **Some of the underlying policy documents genuinely disagree with
   each other**, independent of any retrieval/metric issue — the
   Faculty Handbook and HoD Handbook both say unutilized vacation leave
   "lapses at the end of the corresponding vacation period," while the
   dedicated Leave Information document says vacation leave lapses "31st
   December," for the same policy. No amount of retrieval tuning fixes
   a question whose true answer depends on which of two contradicting
   source PDFs is authoritative — this is a data-quality issue to flag
   to whoever owns those documents, not a code fix.
6. **Re-run `make test-ragas` and `make test-retrieval-eval`** on the
   current config — the committed `ragas_results.json` and
   `retrieval_results.json` predate `retrieval.top_k` moving from 5 to
   8, `mapped_document_primary_top_k` moving from 2 to 3, and the Leave
   topic `keywords` fix (added after the Leave topic's name alone
   scored only 0.103 similarity against real leave questions, well
   under `topic_match_threshold: 0.5`).
