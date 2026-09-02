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

**Query time** (every question typed into the app):

```
user question
    -> retriever.py     (embed the question, fetch top-k nearest chunks from Chroma,
                          drop anything below the relevance threshold)
    -> rag_pipeline.py  (if nothing relevant survived: refuse, no LLM call.
                          otherwise: build a context block from the retrieved chunks
                          and send it + a strict system prompt to Ollama)
    -> llm_client.py    (Ollama chat call)
    -> rag/dashboard.py (render the answer + a link to the single most relevant
                          source PDF)
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
| 1. Document retrieval (load PDFs) | `rag/document_loader.py` |
| 2. Chunking (recursive, URL-safe) | `rag/chunker.py` |
| 3. Sentence embeddings | `rag/embeddings.py` |
| 3. Vector storage (Chroma) | `rag/vector_store.py` |
| 4. Retrieval from Chroma | `rag/retriever.py` |
| 5. Session & chat history (MySQL, 5 min timeout) | `rag/session_store.py` |
| — Email validation (identifies each user) | `rag/user_identity.py` |
| 6. Ollama chat client | `rag/llm_client.py` |
| 7. Grounded answer orchestration | `rag/rag_pipeline.py` |
| Frontend dashboard (all Streamlit UI code) | `rag/dashboard.py` |
| Configuration | `config.yaml`, loaded by `rag/config.py` |

`scripts/ingest.py` runs steps 1-3 to (re)build the vector index.
`scripts/download_embedding_model.py` pre-fetches the embedding model.
`rag/dashboard.py` is the only module with Streamlit UI/dashboard
code; it wires the backend modules above together and holds nothing
that counts as RAG logic itself. `app.py` at the project root is just
the thin entry point Streamlit runs (`streamlit run app.py`), which
imports and calls `rag/dashboard.py`'s `main()`.

## Design decisions

| Decision | Why |
|---|---|
| One module per RAG step | Each stage (load, chunk, embed, store, retrieve, generate) can be read, tested, and replaced on its own without touching the others. |
| Local Ollama (`llama3.2:3b`) for generation | Runs fully offline/on-prem — no policy data leaves the machine. The model is a config value (`ollama.model`), not hard-coded, so swapping to a bigger/smaller/different model later is a one-line change. Started with `llama3.1:8b`; switched to `llama3.2:3b` because this machine runs Ollama on CPU only (inside WSL2, no GPU) and the smaller model answers noticeably faster while still following the strict grounding/refusal instructions reliably. |
| Local `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings, not an Ollama embedding model | Keeps embedding independent of whatever LLM is configured, runs in-process (no extra network hop), and is small/fast enough to embed every chunk and every query on CPU. |
| Recursive, separator-cascade chunker (`chunker.py`) instead of a fixed character splitter | Tries paragraph breaks first, then line breaks, then sentence stops, and only hard-cuts by character count as a last resort — so a chunk boundary lands on a natural break in the text instead of mid-sentence whenever possible. |
| `chunk_size: 800`, `chunk_overlap: 120` (characters) | 800 characters is roughly one dense policy paragraph — big enough to keep a rule and its conditions together, small enough that 4 retrieved chunks still fit comfortably in the LLM's context window. The 120-character overlap (~15%) carries the tail of one chunk into the next so a sentence split across a chunk boundary isn't lost to either side. |
| URLs are masked before sentence-stop splitting | Policy PDFs contain reference URLs (e.g. login portals). Splitting naively on `". "` would cut a URL in half at any `.` in the domain/path. The chunker temporarily replaces each URL with a placeholder, splits, then restores the URL — so links always survive intact in a single chunk. |
| `retrieval.top_k: 4` | Enough chunks to cover a policy that spans a couple of paragraphs or pages, without diluting the LLM's context with too much marginally-relevant text. |
| `retrieval.min_relevance_score: 0.35` (cosine similarity) | Chunks below this score are dropped before the LLM ever sees them. If *nothing* survives the filter, the pipeline refuses immediately without an LLM call — this is the main mechanism that keeps the assistant from answering off-topic questions using the model's own general knowledge. |
| `temperature: 0` | Deterministic, repeatable answers for the same question — appropriate for a policy lookup tool where consistency matters more than creative phrasing. |
| Strict system prompt + context-only instruction (`rag_pipeline.py`) | Second layer of grounding: even when relevant chunks are found, the LLM is explicitly told to answer only from the supplied context and to say so if the context doesn't actually answer the question. |
| `ollama.num_thread: 8`, set explicitly | On this machine (Ollama on CPU inside WSL2), Ollama's thread auto-detection undershot the available 8 cores. Pinning `num_thread` to the real core count (`nproc`) cut a real test query from ~110s to ~55s. Passed through per-request via `options.num_thread` in `llm_client.py`; adjust to match whatever machine runs this. |
| Q4_K_M-quantized models, not fp16 | Both `llama3.1:8b` and `llama3.2:3b` are pulled from Ollama's default tags, which are already 4-bit quantized GGUF (`ollama show <model>` confirms `quantization Q4_K_M`) — several times smaller and faster on CPU than an fp16 checkpoint, with an acceptable quality trade-off for grounded Q&A. Nothing to change here; noted so it isn't accidentally undone by pulling a different/fp16 tag later. |
| Embeddings generated in batches (`embedding.batch_size: 64`) | `embed_texts()` passes the whole chunk list to `sentence-transformers` in one call, which internally batches it (`SentenceTransformer.encode(..., batch_size=...)`) instead of embedding one chunk at a time — reclaims CPU throughput during ingestion essentially for free, since `all-MiniLM-L6-v2` is small enough to batch comfortably. |
| Chroma, local/persistent (not an in-memory or hosted vector DB) | Matches the "local chroma db" requirement; index survives app restarts; no external service dependency. |
| `chromadb==1.5.9` pinned in `requirements.txt` | This machine's `chroma --version` CLI reports `"1.4.4"`, but that's a version-string quirk in this chromadb release's CLI entry point — `chromadb.__version__` and `pip show chromadb` both report `1.5.9`, and there is no `chromadb==1.4.4` published on PyPI to pin to. Pinned to `1.5.9`, the version actually installed and smoke-tested (`PersistentClient` init, collection reload, and query all verified against it), so a future `pip install` can't silently drift to an incompatible release. |
| Streamlit resource caching (`@st.cache_resource` in `app.py`) | Streamlit reruns the whole script on every interaction. `get_config()`, `get_rag_pipeline()` (embedding model + Chroma client + Ollama client), and `get_session_store()` are all wrapped in `@st.cache_resource`, so they're built once per server process and reused for every keystroke/click/query — not reloaded on every rerun. This only saves setup cost, not per-query inference time. |
| MySQL for sessions instead of only in-memory Streamlit state | Streamlit's `session_state` is per-browser-tab and lost on refresh/restart; MySQL gives durable session/chat history across restarts and a shared store that scales to the expected <50 concurrent users. |
| Sessions identified by email, not anonymous UUID only | Ties conversation history in MySQL back to a real person (`rag/user_identity.py` validates the address) rather than being fully anonymous. |
| 5-minute inactivity timeout | Chosen per requirement; implemented as "no query in the last 5 minutes" (`sessions.last_active_at`), not a hard session lifetime — an actively-chatting user is never cut off. |
| Answer shows only the single top-ranked source, as a download link | Chroma already returns results ordered by relevance, so the first item is the best match; showing one clear source (with a real link to open the PDF) is easier to verify than a dropdown of four partially-relevant citations. |
| No lambdas; plain functions/classes everywhere | Readability requirement — every transformation has a name and can be stepped through or unit-tested directly. |
| Configuration centralized in `config.yaml` | Every tunable (model names, chunk size/overlap, top-k, relevance threshold, timeouts, DB connection) lives in one file, loaded by `rag/config.py` into typed dataclasses — no magic numbers scattered through the code. |

## Setup

```bash
make install
```

Make sure Ollama is running locally with the configured model pulled:

```bash
ollama pull llama3.2:3b
```

Download/verify the local embedding model:

```bash
make download-embedding-model
```

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

For verbose Streamlit logging while troubleshooting:

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
| `make run-debug` | Starts the Streamlit app with `--logger.level=debug` and an in-app debug panel (query, retrieved chunks per document, similarity scores) |
| `make test` | Runs the test suite |

## Configuration

All tunables live in `config.yaml`, loaded by `rag/config.py` into
typed dataclasses — no code changes needed to retune the system:

| Section | Key | Current value | Meaning |
|---|---|---|---|
| `ollama` | `base_url` | `http://localhost:11434` | Ollama server address |
| `ollama` | `model` | `llama3.2:3b` | Generation model; swap to any pulled Ollama model |
| `ollama` | `temperature` | `0` | Sampling temperature — `0` for deterministic answers |
| `ollama` | `request_timeout_seconds` | `300` | HTTP timeout for a generation call (CPU inference can be slow) |
| `ollama` | `num_thread` | `8` | CPU threads Ollama uses per generation call; set to your machine's core count (`nproc`) |
| `embedding` | `model_name` | `all-MiniLM-L6-v2` | Local sentence-transformers model for chunk/query embeddings |
| `embedding` | `batch_size` | `64` | Chunks embedded per batch during ingestion |
| `chroma` | `persist_directory` | `./chroma_store` | On-disk location of the vector index |
| `chroma` | `collection_name` | `college_policies` | Chroma collection name |
| `documents` | `source_directory` | `./policies_docs` | Where source PDFs are read from |
| `chunking` | `chunk_size` | `800` (characters) | Target chunk size |
| `chunking` | `chunk_overlap` | `120` (characters) | Overlap carried between consecutive chunks |
| `chunking` | `separators` | `["\n\n", "\n", ". ", " "]` | Split priority order (paragraph → line → sentence → word) |
| `retrieval` | `top_k` | `4` | Number of chunks retrieved per question |
| `retrieval` | `min_relevance_score` | `0.35` | Minimum cosine similarity to keep a chunk; below this, refuse instead of guessing |
| `session` | `timeout_minutes` | `5` | Inactivity timeout before a session is treated as expired |
| `database` | `host` / `port` / `user` / `database` | `127.0.0.1` / `3306` / `sirkumar` / `rag_sessions` | MySQL connection for session/chat history |
| `database` | `password` | — | Can be overridden with the `RAG_DB_PASSWORD` env var instead of editing the file |

## Tests

```bash
make test
```

Tests that need a live MySQL server are skipped automatically if one
is not reachable. Tests that need a live Ollama server are not
included, since generation is exercised through the app itself;
`rag_pipeline` and `retriever` are tested with fakes instead of real
network calls.
