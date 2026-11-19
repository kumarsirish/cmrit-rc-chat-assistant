# Design

This documents the current architecture and the reasoning behind its
notable decisions — what exists in the code today, not the history of
how it got there. See `README.md` for setup/usage and a config
reference.

## Goals

- Answers must be grounded strictly in the indexed documents — refuse
  rather than guess when nothing relevant is found.
- Every tunable (models, thresholds, chunk size, timeouts, DB
  connection) lives in `config/config.yaml`, not scattered through code.
- Each pipeline stage is its own module, independently readable,
  testable, and swappable.
- An admin can add/replace/retire documents and rebuild the index
  entirely through the UI — no command line needed for day-to-day
  document management.
- Every query is traceable after the fact: a bad answer should be
  explainable from a log file alone, without needing to reproduce it
  live.

## Pipeline overview

```
knowledge_documents/ --[ingest]--> Chroma + BM25-ready chunks
                                          |
                      user question ------+
                                          v
                         retrieval_service.py (topic match -> PRIMARY/
                         SECONDARY/fallback hybrid search -> rerank ->
                         relevance filter)
                                          v
                         rag_pipeline.py (refuse, or build context +
                         call the LLM)
                                          v
                         response_processor.py (citation validation,
                         source mapping, grounded/low-confidence flags)
                                          v
                         query_logger.py (JSON trace) + dashboard.py (render)
```

Ingestion and query-time retrieval are deliberately separate concerns
reading/writing the same two stores (Chroma, `config/documents.json`):
ingestion never runs on the request path except when explicitly
triggered (`make ingest` or the Admin UI's "Rebuild index" button, both
calling the same `helper_scripts/ingest.py`).

## Document metadata: one file, not two

`config/documents.json` (`rag/_03_routing/document_metadata.py`) is the
single source of truth for per-document bookkeeping: filename,
category, upload time, active/archived status and supersession
history, plus the topic-matching fields (canonical link, optional
display-name override, curated keywords).

This merges what used to be two separately-maintained files — a
topic-centric `docs-metadata.json` and an upload-centric
`document_registry.json` — because they were already keyed 1:1 by
filename, and a topic's display name was always just the filename with
the extension stripped. Keeping them as two files with two write paths
meant every upload/supersede operation had to update both consistently
by hand; one file removes that failure mode entirely.

`status` is the pivot the rest of the system respects: `"active"`
records are searchable, `"archived"` ones (superseded, physically
moved into an `_archived` subfolder) are kept for audit/history but
must never become routable again. Exactly one function —
`active_doc_topics()` — filters on status; every retrieval-facing
caller goes through it rather than re-implementing the filter, so a
caller forgetting to check status can't accidentally resurrect an
archived document.

## Retrieval policy: FOCUSSED_CATEGORY / GLOBAL, PRIMARY / SECONDARY / fallback

`rag/_04_retrieval/retrieval_service.py` is the only module aware of
topics, categories, or this policy — `dense_retriever.py`,
`bm25_retriever.py`, `hybrid_retriever.py`, and `reranker.py` are generic
mechanisms it calls into.

**Why topic-matching restricts the search scope instead of boosting a
score afterward.** An earlier design boosted a topic-matched chunk's
score *after* a global search, but reranking then fully rescored the
boosted pool and could still bury a correct match under an unrelated
document — the boost only affected who got a chance in front of the
reranker, not who won. Restricting the search itself to the matched
document(s) removes cross-document competition entirely: nothing
(fusion, reranking, or the relevance threshold) can substitute a
different document once one is trusted.

**Why a topic can match more than one document.** Earlier, topic
matching picked a single best-scoring topic and assumed one topic meant
one document. That breaks once a category can hold several related
documents (two different forms filed under the same category, for
example) — a query can legitimately be about more than one of them.
`EmbeddingTopicMatcher.match_all()` returns every topic clearing
`topic_match_threshold`, not just the best one, and PRIMARY searches
the union of all their documents.

**Why FOCUSSED_CATEGORY exists at all, instead of always searching
globally.** When a user has already told the system which category a
question belongs to, trusting that signal and topic-matching only
within it is strictly more informative than a blind corpus-wide search
— fewer, more relevant competing documents means less chance of a
wrong document's chunk outscoring the right one. There is no automatic
category *detection* (no classifier guesses a category when none is
given) — a query is always either scoped to a category the user
explicitly picked, or fully global. This was a deliberate
simplification: an automatic classifier was tried and removed, since it
added a second heuristic (on top of topic matching) that could itself
be wrong, with no way for the user to override a bad guess the way
picking "All categories" lets them override nothing at all.

**Why PRIMARY is a trusted pool, judged by dense score, not reranker
score.** A cross-encoder reranker can genuinely score a correct chunk
low — a short section heading, for instance, carries little semantic
signal on its own even though it's the right chunk. Gating PRIMARY on
reranker score would let that single bad score evict the mapped
document's own best chunk and force a worse fallback. The reranker
still runs over PRIMARY when configured, but only to *reorder* the
already-trusted pool, never to drop a chunk from it outright.
SECONDARY and the fallback search are not from a trusted pool, so
reranker score stays the hard gate there.

**Why PRIMARY falls back to a flat search when it comes up empty.**
Topic matching is a similarity heuristic and can still pick the wrong
document. Trusting a wrong match with no escape hatch would turn a bad
topic guess into a hard refusal; falling back to an ordinary hybrid
search over the same scope only when the trusted pool is genuinely
empty keeps the common case strict without making a heuristic's
mistake fatal.

**Why hybrid (dense + BM25) rather than dense-only.** Dense embeddings
miss exact acronyms, policy codes, and proper nouns the model has too
little signal to place correctly; BM25 alone misses paraphrases dense
search catches naturally. Reciprocal Rank Fusion combines the two
ranked lists by rank rather than raw score (cosine similarity and BM25
scores aren't on a comparable scale), so whichever signal is actually
strong for a given query carries the result without the app having to
guess in advance which one that is.

**Why reranking is a second, optional pass over a small pool, not the
primary filter.** A cross-encoder scoring `(query, chunk)` jointly is a
strictly stronger relevance signal than comparing two independently-
computed embeddings, but too slow to run over the whole collection — so
it only rescores the fused pool's best `rerank.candidates`, and that
pool size is deliberately kept larger than `hybrid.candidates` so a
chunk that dense/BM25 already found isn't silently dropped — by RRF
rank, not true relevance — before the reranker ever sees it.

**Why a low-confidence fallback instead of a flat refuse/answer
binary.** Below `min_reranker_score`, the single best-scoring candidate
is still returned — flagged `low_confidence`, shown to the user with an
explicit caveat — as long as it clears a much lower floor
(`low_confidence_min_score`). A truly unrelated question scores near
zero against the corpus and still gets a clean refusal; a genuine near-
miss (real content, just scored lower than the normal bar) gets
surfaced instead of silently discarded, with the UI making clear it
should be double-checked.

## Generation and grounding

Two layers keep the assistant from answering outside the retrieved
text, enforced in `rag/_05_generation/rag_pipeline.py`:

1. If retrieval finds nothing above the relevance threshold, the
   pipeline refuses immediately without an LLM call at all.
2. If chunks are found, the LLM gets a strict system prompt restricting
   it to the supplied numbered context and told to reply with an exact,
   checkable refusal sentence if the context doesn't answer the
   question — `response_processor.py` checks for that exact sentence to
   decide `grounded`.

Citations are enforced structurally, not just by instruction:
`response_processor._normalize_citations` rewrites any verbose or
prose-style citation the model produces into the short `[N]` form, then
strips any `[N]` whose N doesn't correspond to a real retrieved chunk —
so a citation left in the final answer is always real, never
fabricated or just plausible-looking.

**Conversation history is passed to the LLM, never to retrieval.**
`conversation_history` lets the LLM resolve references like "it"/"that"
in its own reply, but retrieval always runs on the literal question
alone. A prior design rewrote a follow-up question into a standalone
retrieval query (query condensation) before searching; it was removed
because the rewritten query could match far more topics than intended
— `match_all()`'s multi-topic admission (see above) amplified that
risk, since a longer, more generic rewritten question had more
surface area to spuriously clear `topic_match_threshold` against
unrelated documents. Retrieval is deterministic and evaluable
independent of session state in the meantime; reintroducing
follow-up-aware retrieval is a deliberate future redesign, not an
oversight.

**Rephrase suggestion is a separate, single-turn mechanism**
(`rephrase_suggester.py`), not to be confused with the conversation-
history mechanism above: when the plain "nothing found" refusal fires,
one extra LLM call asks for a HyDE-style rewrite — not a paraphrase of
the question, but a guess at how the *answer* would actually be worded
in a policy document — and offers it as a one-click retry. It never
feeds back into retrieval automatically, so one no-result answer can't
silently chain into unseen retries.

## Observability

Every query writes one JSON line to a per-process log file
(`rag/_09_common/query_logger.py`), covering the full pipeline: the
question, every stage's fused candidates (dense/lexical/fused scores),
every reranker score (not just the chunks that survived), the final
kept chunks, the LLM's raw reply, and per-stage timings. The intent is
that a bad answer, found later, can be traced back to exactly which
stage produced it — a wrong document reaching PRIMARY, a correct chunk
scored too low by the reranker, a citation the LLM invented — from the
log alone, without needing to reproduce the question live.

Design choices behind it:

- **JSON Lines, not a single JSON array.** Each line is independently
  valid and the file can be safely appended to while the app keeps
  running, with no array-closing bookkeeping that a crash could leave
  broken.
- **A new file per process launch**, timestamped at import time (module
  top-level code runs once per process) — restarting the app never
  appends to or overwrites a previous run's log.
- **Injected as a callback, defaulting to a no-op.** `RagPipeline`
  accepts an optional `log_query` function; only
  `main_pipeline.build_rag_pipeline()` (the one place the live app,
  evaluation scripts, and experiment runs all assemble the pipeline
  from) wires in the real file-writing function. Tests that construct
  `RagPipeline` directly get the no-op by default, so the test suite
  never writes to the real `logs/` directory.
- **Independent of `RAG_DEBUG`.** The JSON log always runs; `RAG_DEBUG`
  only toggles a separate in-app expander showing a lighter-weight view
  of the same kind of information, live, in the browser. They don't
  share a code path — removing or changing one doesn't affect the
  other.
- **Retrieval exposes its trace via a `last_trace` attribute, not a
  changed return signature.** `RetrievalService.retrieve()` still
  returns `(chunks, matched_topics)` exactly as before; the fuller trace
  is stashed on `self.last_trace` (overwritten each call) for whichever
  caller wants it. This kept every existing caller (tests, evaluation
  scripts) unchanged while still giving `rag_pipeline.py` everything it
  needs to build the combined log record.

## Session, auth, and chat UX

- **Sessions are identified by email, not an anonymous UUID.**
  `user_identity.py` validates the address so MySQL-stored conversation
  history ties back to a real person rather than being fully anonymous
  — there is no real login system, so this is a traceability choice,
  not a security boundary.
- **MySQL over in-memory Streamlit state.** `st.session_state` is
  per-browser-tab and lost on refresh/restart; MySQL gives durable
  session/chat history across restarts and a shared store across the
  expected handful of concurrent users, plus the data backing the
  Admin tab's cross-user "Recent Queries" view and average-response-
  time stat.
- **5-minute inactivity timeout, not a hard session lifetime.**
  Implemented as "no query in the last N minutes," so an actively-
  chatting user is never cut off mid-conversation; a live countdown
  warns before it fires.
- **The category picker widens automatically on a miss.** Picking a
  specific category and getting an ungrounded answer triggers one
  automatic retry against "All categories" before the UI shows a
  refusal — the first (unanswered) attempt is discarded in favor of
  whatever the broader search finds, rather than making the user
  manually switch and re-ask.
- **Admin rebuild clears the process-wide Streamlit resource cache.**
  `RetrievalService`, the category list, and the BM25 index are all
  built once per process (`@st.cache_resource`) for performance. A
  category or document added, renamed, or removed by a rebuild
  wouldn't take effect until the process restarted otherwise — the
  rebuild's background worker explicitly calls `st.cache_resource.clear()`
  on success so the very next chat interaction, in any session, picks
  up what's now on disk.

## Evaluation

Two separate suites, deliberately not merged into one:

- **Retrieval-only IR metrics** (`test_retrieval_eval.py` —
  Recall/Precision/MRR/NDCG@k against ground-truth relevant sources, no
  LLM call) — cheap and fast, so retrieval regressions are caught
  without paying for generation or a judge model.
- **DeepEval judge** (`test_judge.py` — faithfulness/relevancy/
  correctness, scored by a separate judge model against real pipeline
  answers) — conflates retrieval and generation quality, but is what
  actually matters to a user. Kept separate from the retrieval metrics
  so a regression can be attributed to "retrieval" or "generation," not
  just "something got worse."

The judge is a different model/provider than whatever answers real
questions, specifically so it never grades its own generations.

MLflow experiment tracking (`helper_scripts/run_experiment.py`) is a
separate concern from the pass/fail `make eval` gate above — it logs
comparable runs over time for deliberate hyperparameter/model
comparisons, entirely local (SQLite + a local artifact directory, no
server), and never touches the judge model.

## Things not done, by design

- **No automatic category detection** — removed in favor of an explicit
  picker (see "Retrieval policy" above).
- **No follow-up-aware retrieval query rewriting** — removed pending a
  redesign (see "Generation and grounding" above); a pronoun-laden
  follow-up currently retrieves on its literal text.
- **No file-locking on `config/documents.json`** — acceptable for a
  single-admin internal tool with no concurrent-writer scenario.
