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
`rag/bm25_index.py` lexical index is rebuilt in-memory from the same
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
                          retrieved chunks and send it + a strict system prompt,
                          via LangChain's ChatOllama, to Ollama)
    -> llm_client.py    (LangChain ChatOllama chat call)
    -> response_processor.py (validate/normalize the model's "[N]" citations,
                          map each retrieved chunk to a source reference)
    -> rag/dashboard.py (render the answer + cited sources, each linking to its
                          source PDF/page)
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
| 1. Document loading (load PDFs) | `rag/document_loader.py` |
| 2. Chunking (recursive, URL-safe) | `rag/chunker.py` |
| 3. Sentence embeddings | `rag/embeddings.py` |
| 3. Vector storage (Chroma) | `rag/vector_store.py` |
| 4a. Lexical (BM25) index | `rag/bm25_index.py` |
| 4b. Cross-encoder reranking | `rag/reranker.py` |
| 4c. Topic detection (query -> docs-metadata.json entry) | `rag/docs_metadata.py`, `rag/keyword_extraction.py` |
| 4. Retrieval (topic-mapped-document or hybrid dense+BM25 fallback + rerank) | `rag/retriever.py` |
| 5. Session & chat history (MySQL, 5 min timeout) | `rag/session_store.py` |
| — Email validation (identifies each user) | `rag/user_identity.py` |
| 6. LLM chat client (LangChain ChatOllama) | `rag/llm_client.py` |
| 6. Context block formatting | `rag/context_builder.py` |
| 6. Citation validation / answer assembly | `rag/response_processor.py` |
| 7. Grounded answer orchestration | `rag/rag_pipeline.py` |
| Frontend dashboard (all Streamlit UI code) | `rag/dashboard.py` |
| Configuration | `config.yaml`, loaded by `rag/config.py` |

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
  `docs-metadata.json` that has a locally-indexed document (keyword-
  overlap match, see `find_matching_topic`). That topic -> document
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
   (`keyword_extraction.py`, stopwords stripped) are matched against
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
5. **Relevance threshold** — anything below `min_relevance_score`
   (dense score, or lexical-only hits, which skip the threshold since
   BM25 isn't on the same 0-1 scale) is dropped before the LLM ever
   sees it. If nothing survives (and, for MAPPED_DOCUMENT mode, the
   GLOBAL fallback also finds nothing), `rag_pipeline.py` refuses
   without calling the LLM at all.

Every query logs one summary line (`rag.retriever` logger, INFO level)
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
| One module per RAG step | Each stage (load, chunk, embed, store, retrieve, generate) can be read, tested, and replaced on its own without touching the others. |
| Hybrid retrieval (dense + BM25 fused by RRF), not dense-only | Dense embeddings miss exact acronyms/policy codes/proper nouns the model has too little signal to place correctly (e.g. "TYL"); BM25 alone misses paraphrases dense search catches naturally. Fusing by rank rather than picking a winner or normalizing scores lets whichever signal is actually strong for a given query carry the result, without the app guessing in advance which one that is. |
| Cross-encoder reranking as an optional second pass | A cross-encoder scoring `(query, chunk)` jointly is a strictly stronger relevance signal than comparing two independently-computed embeddings, but too slow to run over the whole collection — so it's applied only to the fused pool's best `rerank.candidates`, not the full corpus. |
| Topic detection from `docs-metadata.json` (`docs_metadata.py`) | The college already maintains a topic -> canonical-URL/filename mapping; reusing it as a lightweight keyword-overlap signal restricts retrieval to the right document for on-topic queries, and lets the app surface a source link even for topics with no locally-ingested PDF. |
| A topic match restricts search to the mapped document (MAPPED_DOCUMENT mode), not an additive score boost | An earlier design boosted a topic-matched chunk's score *after* global search, but reranking then fully rescored the pool and could still bury a correct topic match under an unrelated document — the boost only affected who got a chance in front of the reranker, not who won. Restricting the search itself to the mapped document removes cross-document competition entirely: nothing (fusion, threshold, or reranking) can substitute a different document once one is trusted. |
| MAPPED_DOCUMENT mode falls back to GLOBAL search if the mapped document has nothing above the relevance threshold | Topic detection is a lightweight keyword-overlap heuristic and can pick the wrong topic (e.g. a generic word like "details" ties across topics). Trusting a wrong mapping with no escape hatch would turn a bad topic match into a hard refusal; falling back to a normal global search only when the trusted document comes up genuinely empty keeps the common case strict without making a heuristic's mistake fatal. |
| Local Ollama for generation, via LangChain's `ChatOllama` | Runs fully offline/on-prem — no policy data leaves the machine. The model is a config value (`ollama.model`), not hard-coded, so swapping it later is a one-line change. LangChain gives a uniform message/chat interface that's also reused by the RAGAS judge harness. |
| Local `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings, not an Ollama embedding model | Keeps embedding independent of whatever LLM is configured, runs in-process (no extra network hop), and is small/fast enough to embed every chunk and every query on CPU. |
| Recursive, separator-cascade chunker (`chunker.py`) instead of a fixed character splitter | Tries paragraph breaks first, then line breaks, then sentence stops, and only hard-cuts by character count as a last resort — so a chunk boundary lands on a natural break in the text instead of mid-sentence whenever possible. |
| URLs are masked before sentence-stop splitting | Policy PDFs contain reference URLs (e.g. login portals). Splitting naively on `". "` would cut a URL in half at any `.` in the domain/path. The chunker temporarily replaces each URL with a placeholder, splits, then restores the URL — so links always survive intact in a single chunk. |
| `temperature: 0` | Deterministic, repeatable answers for the same question — appropriate for a policy lookup tool where consistency matters more than creative phrasing. |
| Strict system prompt + inline `[N]` citations (`rag_pipeline.py`, `response_processor.py`) | Second layer of grounding on top of the relevance-threshold refusal: the LLM is told to answer only from the supplied numbered context, cite every claim by source number, and say so explicitly if the context doesn't answer the question. `response_processor.py` then normalizes/strips any citation the model invents or writes in prose instead of `[N]`, so a citation left in the final answer is always real. |
| Conversation history passed to the LLM but never to retrieval | Retrieval always runs on the current question alone, so it stays deterministic and easy to evaluate independent of session state; history is only given to the LLM so it can resolve references like "it"/"that leave type" in its reply. |
| `ollama.num_thread: 8`, set explicitly | On this machine (Ollama on CPU inside WSL2), Ollama's thread auto-detection undershot the available 8 cores. Pinning `num_thread` to the real core count (`nproc`) noticeably cut generation latency. Passed through per-request via `ChatOllama`'s `num_thread`; adjust to match whatever machine runs this. |
| Chroma, local/persistent (not an in-memory or hosted vector DB) | Matches the "local chroma db" requirement; index survives app restarts; no external service dependency. |
| BM25 index rebuilt in-memory at startup, not persisted separately | The corpus is small (a few hundred chunks), so `BM25Okapi` over every chunk in the Chroma collection builds fast enough to redo on every process start rather than maintaining a second on-disk index that could drift from Chroma. |
| MySQL for sessions instead of only in-memory Streamlit state | Streamlit's `session_state` is per-browser-tab and lost on refresh/restart; MySQL gives durable session/chat history across restarts and a shared store that scales to the expected <50 concurrent users. |
| Sessions identified by email, not anonymous UUID only | Ties conversation history in MySQL back to a real person (`rag/user_identity.py` validates the address) rather than being fully anonymous. |
| 5-minute inactivity timeout | Chosen per requirement; implemented as "no query in the last 5 minutes" (`sessions.last_active_at`), not a hard session lifetime — an actively-chatting user is never cut off. |
| Two separate evaluation suites, not one | `test_retrieval_eval.py` scores the retriever alone (classic IR metrics — Recall/Precision/MRR/NDCG@k) with no LLM call, so retrieval regressions are cheap and fast to catch. `test_ragas_judge.py` scores the full pipeline's generated answers with an external judge model (faithfulness/relevancy/correctness), which conflates retrieval and generation quality but is what actually matters to a user. Keeping them separate makes it possible to tell "retrieval is fine, generation is the problem" apart from the reverse. |
| RAGAS judge is Gemini (`gemini-2.5-flash`), not the same Ollama model that answers | A judge grading its own generations is a biased signal (it tends to rate its own phrasing/reasoning favorably). Gemini also runs faster/more capably than the local CPU-bound Ollama model would for a judge role. |
| No lambdas; plain functions/classes everywhere | Readability requirement — every transformation has a name and can be stepped through or unit-tested directly. |
| Configuration centralized in `config.yaml` | Every tunable (model names, chunk size/overlap, retrieval/rerank/hybrid settings, evaluation thresholds, timeouts, DB connection) lives in one file, loaded by `rag/config.py` into typed dataclasses — no magic numbers scattered through the code. |

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

### 3. Ollama

Make sure Ollama is running locally with the configured model pulled:

```bash
ollama pull llama3.1:8b
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

## Configuration

All tunables live in `config.yaml`, loaded by `rag/config.py` into
typed dataclasses — no code changes needed to retune the system:

| Section | Key | Current value | Meaning |
|---|---|---|---|
| `ollama` | `base_url` | `http://localhost:11434` | Ollama server address |
| `ollama` | `model` | `llama3.1:8b` | Generation model; swap to any pulled Ollama model |
| `ollama` | `temperature` | `0` | Sampling temperature — `0` for deterministic answers |
| `ollama` | `request_timeout_seconds` | `300` | HTTP timeout for a generation call (CPU inference can be slow) |
| `ollama` | `num_thread` | `8` | CPU threads Ollama uses per generation call; set to your machine's core count (`nproc`) |
| `embedding` | `model_name` | `all-MiniLM-L6-v2` | Local sentence-transformers model for chunk/query embeddings |
| `embedding` | `batch_size` | `64` | Chunks embedded per batch during ingestion |
| `chroma` | `persist_directory` | `./chroma_store` | On-disk location of the vector index |
| `chroma` | `collection_name` | `college_policies` | Chroma collection name |
| `documents` | `source_directory` | `./policies_docs` | Where source PDFs are read from |
| `documents` | `metadata_file` | `./docs-metadata.json` | Topic -> canonical URL/filename lookup, used to restrict retrieval to a topic's mapped document (MAPPED_DOCUMENT mode) and for source links |
| `chunking` | `chunk_size` | `800` (characters) | Target chunk size — roughly one dense policy paragraph: big enough to keep a rule and its conditions together, small enough that several retrieved chunks still fit comfortably in the LLM's context window |
| `chunking` | `chunk_overlap` | `120` (characters, ~15%) | Overlap carried between consecutive chunks, so a sentence split across a boundary isn't lost to either side |
| `chunking` | `separators` | `["\n\n", "\n", ". ", " "]` | Split priority order (paragraph -> line -> sentence -> word) |
| `retrieval` | `top_k` | `3` | Final chunks returned per question, in both MAPPED_DOCUMENT and GLOBAL mode, after fusion/rerank/filtering |
| `retrieval` | `min_relevance_score` | `0.40` | Minimum dense cosine similarity to keep a pure-dense chunk; lexical-only hits skip this (different scale). If nothing survives, the pipeline refuses instead of guessing |
| `retrieval.hybrid` | `enabled` | `true` | GLOBAL mode only: turns on BM25 lexical search fused with dense search via RRF; disabled means dense-only |
| `retrieval.hybrid` | `candidates` | `10` | GLOBAL mode: chunks fetched from *each* of dense/lexical search before fusion. MAPPED_DOCUMENT mode uses a separate, smaller constant (`MAPPED_DOCUMENT_CANDIDATES = 6` in `retriever.py`, not config — the search space is already one document) |
| `retrieval.rerank` | `enabled` | `true` | Turns on the cross-encoder reranking pass over the candidate pool (fused GLOBAL pool, or the small MAPPED_DOCUMENT pool) |
| `retrieval.rerank` | `model` | `BAAI/bge-reranker-base` | Cross-encoder model (alt: `ms-marco-MiniLM-L-6-v2`) |
| `retrieval.rerank` | `candidates` | `8` | Top N of the candidate pool handed to the reranker; the final `top_k` is chosen from these |
| `session` | `timeout_minutes` | `5` | Inactivity timeout before a session is treated as expired |
| `session` | `max_history_messages` | `2` | Most recent chat turns sent to the LLM as conversation context |
| `evaluation` | `judge_provider` / `judge_model` | `gemini` / `gemini-2.5-flash` | External judge for `test-ragas`, kept separate from `ollama.model` so it isn't grading its own generations |
| `evaluation` | `test_data_path` | `./tests/test-question-qnswers.json` | Question/reference-answer (+ optional `relevant_sources`) test cases used by both eval suites |
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
need a live Ollama server and/or an ingested Chroma store and are
excluded from `make test` by default (`pytest.ini`); run them via
`make test-retrieval-eval` / `make test-ragas` / `make metrics`.
`rag_pipeline` and `retriever` unit tests use fakes instead of real
network calls.

## Current metrics — what they say, and what's still weak

`make metrics` writes fresh numbers to `tests/retrieval_results.json`
(classic IR) and `tests/ragas_results.json` (LLM-judged), both also
readable from the app's System Info page.

### Retrieval-only (`test-retrieval-eval`, k=3, 4 labeled questions)

Retrieval was reworked since the numbers above were first recorded:
a query that matches a topic in `docs-metadata.json` with a
locally-indexed document (MAPPED_DOCUMENT mode) now searches *only*
that document, instead of searching everything and hoping an additive
score boost plus reranking would surface it — see "Retrieval, in
detail" above. Current numbers on the same 4-question set:

| Category | Recall@3 | Precision@3 | MRR | NDCG@3 |
|---|---|---|---|---|
| `normal` (3 questions) | 0.50 | 0.44 | 0.67 | 0.91 |
| `multiple_related_documents` (1 question) | 0.00 | 0.00 | 0.00 | 0.00 |
| **Overall mean** | **0.375** | **0.33** | **0.50** | 0.69 |

**This now fails the configured minimums** (`min_recall: 0.6`,
`min_mrr: 0.5`) — recall got worse, not better, on this ground truth.
That's an expected, direct consequence of the redesign, not a bug, and
it's worth understanding exactly why before deciding what to do about
it:

- **"what is tyl" (recall 0.5, was 0.5)** — now correctly resolves to
  MAPPED_DOCUMENT mode and returns 3/3 chunks from
  `TYL - Faculty Orientation...pdf`, the topic's authoritative
  document (previously the reranker was dropping it entirely — see
  git history). Recall is still only 0.5 because the ground truth in
  `test-question-qnswers.json` lists *two* `relevant_sources`
  (`TYL - Faculty Orientation...pdf` **and** `HoD Handbook - CMRIT.pdf`,
  which mentions TYL only in passing). MAPPED_DOCUMENT mode deliberately
  never returns a second document once one is trusted, so this ground
  truth and the new design are structurally in tension — see item 1
  below.
- **"Give TYL Program Details" (recall 0.0, was 0.5)** — a real
  regression. `find_matching_topic`'s keyword-overlap scoring ties
  between the correct "TYL Process - Training and Placement" topic and
  "Faculty Handbook (...Contact Details etc.)" (both share exactly one
  keyword with the query — "tyl" vs. "details"), and its tie-break
  ("prefer the longer topic string") picks Faculty Handbook. That
  document has nothing relevant, so MAPPED_DOCUMENT mode correctly
  falls back to GLOBAL — but the global fallback then landed on
  `Student Handbook-CMRIT.docx.pdf`, not either expected document. This
  is a pre-existing weakness in `find_matching_topic`'s tie-breaking
  (generic words like "details" count as much as "tyl"), just newly
  consequential now that a topic match determines the entire search
  scope. See item 2 below.
- **"how to get vacation leave approved" (recall 1.0, was 1.0)** and
  the rest of `normal` are unaffected or improved — single-document,
  unambiguous topic matches are exactly the case MAPPED_DOCUMENT mode
  is built for.

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
   `test-question-qnswers.json` lists `HoD Handbook - CMRIT.pdf` as a
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
2. **`find_matching_topic`'s tie-breaking picked the wrong topic for
   "Give TYL Program Details"** — it ties on keyword-overlap count
   between the correct TYL topic and an unrelated one sharing only a
   generic word ("details"), and its tie-break (longer topic string
   wins) doesn't favor the more specific match. This was always a
   latent weakness, but restricting retrieval to the resolved topic's
   document makes a wrong resolution far more consequential than
   before (it now determines the entire search scope, not just a soft
   boost). Worth either weighting rarer/more specific keywords more
   than common ones, or requiring a minimum overlap fraction (not just
   count) before trusting a topic match enough to restrict search.
3. **The GLOBAL fallback (for a bad topic match, or no match at all)
   still has the pre-existing weaknesses hybrid+rerank always had** —
   e.g. no mechanism to diversify across source documents for a
   question that genuinely spans more than one policy doc (the
   `multiple_related_documents` category). MAPPED_DOCUMENT mode doesn't
   touch this; it only helps when a topic resolves correctly to one
   document.
4. **Test set size** — 4 labeled retrieval cases (1 per category
   except `normal`) means every metric is dominated by one or two
   questions; before trusting any further tuning change, the test set
   needs more labeled cases per category so the mean isn't noise, and
   ideally more single-topic MAPPED_DOCUMENT-shaped questions to
   validate the new path specifically.
5. **Re-run `make test-ragas`** on the current retrieval code — the
   committed `ragas_results.json` predates this refactor.
