# CMRIT College Policy RAG Assistant — System Design Document

Retrieval-Augmented Generation pipeline for answering staff and student questions from CMRIT's policy documents.

## 1. System Overview

This document describes the design of the CMRIT College Policy RAG Assistant — a chatbot that answers questions about college policies (examinations, leave, probation, and similar topics) by reading the actual policy documents, not by guessing from general knowledge.

The system is built as a RAG (Retrieval-Augmented Generation) pipeline. In simple terms: instead of asking a language model to answer directly, the system first **retrieves** the exact passages from the policy documents that are relevant to the question, and then asks the language model to **generate** an answer using only those passages. This keeps the assistant's answers grounded in real college policy and lets it cite the exact source and page for every claim.

The architecture has five parts:

- An **offline ingestion pipeline** that reads source documents and builds a searchable index.
- A **query understanding layer** that cleans up and disambiguates a raw question — resolving follow-up references and institutional jargon — before it ever reaches search.
- An **online retrieval and routing layer** that decides which documents are relevant to a question and finds the best passages within them.
- A **response generation layer** that turns retrieved passages into a cited, grounded answer.
- **Supporting infrastructure** — session/conversation persistence, caching, configuration, and an evaluation/experimentation practice — that the other four rely on.

The rest of this document walks through each stage in that order, followed by how the system is tested and tuned, and a short glossary of the underlying techniques for anyone new to RAG.

## 2. Ingestion Pipeline

Ingestion is the **offline** half of the system. It runs once, and again any time a policy document is added, removed, or changed. Its job is to turn raw documents into a searchable index that the online pipeline can query in milliseconds.

In order: the **Document Loader** extracts text page by page from each source document; a **Deduplication** pass drops near-identical repeated content; the **Chunker** splits the remaining text into short, overlapping passages small enough for the embedding model and the language model to work with; a **Glossary Builder** separately mines the same documents for institutional acronyms; the **Embedding Generator** turns each chunk into a numeric vector that captures its meaning; and the result is written into a vector store on disk.

### 2.1 Hyperparameters and Models

| Setting | Value | Purpose |
|---|---|---|
| Embedding model | A mid-sized local sentence-embedding model | Turns each chunk and each query into a vector for meaning-based (semantic) search. Chosen for stronger paraphrase matching than a smaller, faster alternative — the assistant has to match conversational questions against formally-worded policy text, where exact wording rarely overlaps. |
| Embedding batch size | 64 | Chunks are embedded 64 at a time instead of one at a time, to use the CPU efficiently during ingestion. |
| Chunk size | 800 characters | Roughly one to two paragraphs — small enough to keep a chunk focused on one idea, large enough to keep a rule and its context together. |
| Chunk overlap | 120 characters | Consecutive chunks share their last 120 characters, so a sentence that falls right on a chunk boundary is not cut in half and lost. |
| Splitting order | Paragraph break → line break → sentence stop → hard character limit | The splitter always tries the most natural break first (a paragraph), and only falls back to a harder cut if a chunk is still too long. |
| Vector store | An embedded, in-process vector database (persisted to a local folder) | No separate server to install or run. |

### 2.2 Deduplication: Why Identical Chunks Are Dropped at Ingestion

Several institutional documents (handbooks for different staff/student categories, for example) share long stretches of identical front matter — an "about the institution" section, trust history, leadership bios — verbatim. Left in the index as-is, that boilerplate gets embedded once per document it appears in, so its vector is reinforced several times over. Because it mentions the institution's name prominently, it scores deceptively well against almost any question that also mentions the institution by name — well enough to crowd out a chunk that is genuinely specific to the question but appears only once in the whole collection.

To prevent this, ingestion normalizes each chunk's text (collapsing whitespace) and drops any chunk whose normalized text has already been seen, keeping only the first occurrence (and that occurrence's own source/page, for citation purposes). This is applied after chunking and before embedding, so duplicate content is never embedded or indexed more than once.

### 2.3 Why Is BM25 Not Persisted, but Rebuilt at App Launch?

BM25 is the keyword-search index used alongside vector (semantic) search — see the Appendix for what BM25 does. Unlike the main vector store, the BM25 index is not saved to disk. It is rebuilt from scratch, in memory, every time the application starts, by reading every chunk back out of the vector store.

This is a deliberate choice, for three reasons:

- **Single source of truth.** The vector store is already the one place chunk text and metadata live. Keeping a second, separately-saved BM25 file risks it going out of sync after a re-ingestion — for example, if ingestion updates the vector store but the old BM25 file is forgotten, searches would return chunks that no longer exist.
- **It's cheap to rebuild.** The whole policy collection is only a few hundred to a few thousand chunks. Building an in-memory keyword index over that many short passages takes a fraction of a second — there is no real cost to rebuilding it every time, unlike, say, retraining a machine learning model.
- **Simplicity.** Not persisting BM25 means there is one less file format to manage, version, and keep consistent with the vector store — fewer moving parts, fewer ways for the two indexes to drift apart.

### 2.4 Glossary Extraction: Bridging Institutional Jargon at Index Time

Policy documents are full of institution-specific acronyms (committee names, process codes) that a question phrased in plain language won't use. Rather than hand-maintain a glossary, the ingestion pipeline automatically mines one from the documents themselves: institutional writing conventionally spells an acronym out on first use, in one of two forms — "full phrase (ACRONYM)" or "ACRONYM (full phrase)". A pattern-matching pass over the ingested text captures both forms and keeps only the matches where the acronym's letters are genuinely the initials of the captured phrase, discarding the much larger set of parenthetical asides that aren't real definitions.

The resulting acronym → phrase glossary is not used to teach the answer-generation model new vocabulary (the pipeline is strictly grounded in retrieved text, so that would let the model reference acronyms with no retrieved content behind them). Instead it is consumed entirely on the retrieval side — see Section 3.5.

### 2.5 Justification for Major Design Decisions

- **A small, CPU-capable embedding model.** The application is designed to run without requiring a GPU. The embedding model is chosen as a balance of speed and search quality for that constraint, rather than the largest or most accurate model available, with paraphrase-matching quality weighted heavily since conversational questions rarely use the policy document's exact wording.
- **A recursive, paragraph-first chunker.** Policy documents are written in numbered clauses and short paragraphs. Splitting on paragraph breaks first (rather than, say, a fixed character count with no regard for sentence boundaries) keeps each chunk readable and self-contained, which directly improves both search relevance and the quality of citations shown to the user.
- **An embedded vector database, not a separate server.** This keeps deployment simple — there is no extra service to install, configure, or keep running alongside the app.
- **Deduplication at ingestion rather than at query time.** Filtering repeated boilerplate out of search results after the fact would require detecting the duplication freshly on every query; removing it once, at index-build time, is both cheaper and guarantees every downstream consumer (dense search, BM25, reranking) sees a clean corpus.

### 2.6 Limitations / Improvements Needed

- **No incremental ingestion.** Re-running ingestion rebuilds the entire index from all documents every time, even if only one changed. For a larger document set, this would become slow and wasteful.
- **No automatic re-ingestion trigger.** Someone has to remember to re-run ingestion after a document changes; there is no file-watcher or scheduled job that does this automatically today.
- **Scanned/image-only documents are skipped, not read.** If a policy document is a scanned image rather than real text, the loader currently logs an error and skips it rather than running OCR — that document is silently absent from the index unless someone checks the ingestion log.
- **The glossary is extracted, not curated.** The acronym-mining pass is a purely mechanical pattern match; it can miss acronyms that are never spelled out in the standard "phrase (ACRONYM)" convention, or in rare cases attach the wrong phrase to an acronym. It isn't reviewed by a person before use.

## 3. Retrieval Design

Retrieval is the online step that decides which policy chunks are relevant to a given question, before the language model ever sees them. Getting this step right matters more than almost anything else in the system — the language model is instructed to answer only from what retrieval hands it, so a retrieval mistake becomes an answer mistake.

Retrieval is built on a standard RAG framework for the generic, reusable pieces — the embedding model, vector search, and the cross-encoder reranker all go through standard components, which keeps those pieces easy to swap or upgrade later. Two pieces are deliberately kept custom instead: BM25 keyword search, and the institution-specific routing policy (MAPPED_DOCUMENT / PRIMARY / SECONDARY / GLOBAL) described below. Off-the-shelf BM25 tools cannot filter results by document after scoring, which the routing policy below depends on, and lack a safeguard against a false "match" with no real word overlap in a small collection; building one separate BM25 index per document instead of one shared index was considered and rejected, since it would quietly change how BM25 scores every word, not just add filtering.

### 3.1 Query Preprocessing: Condensation and Expansion

Before a question reaches topic matching or search, it passes through two independent, narrowly-scoped rewriting steps. Both operate only on the text used for retrieval — the original question is always what is shown to the user and sent to the answer-generation model, so citations and phrasing are never affected by either step.

**Follow-up condensation.** A genuine multi-turn follow-up ("How many days is that?") embeds and searches to nothing useful on its own, since dense/BM25 retrieval only ever sees the text it is given directly — unlike the answer-generation model, it cannot resolve a pronoun by reading conversation history itself. To bridge that gap, a single model call looks at only the single most recent exchange in the conversation (never the full session history) and decides whether the new question depends on it. If it does, the call rewrites the question into a standalone form that spells out the specific detail being referred to; if it already stands on its own — including about a brand-new, unrelated topic — it is returned unchanged, verbatim. Any failure of this call (an error, an empty reply) degrades gracefully to using the original, unmodified question, rather than blocking the turn.

**Glossary expansion.** Separately, the (possibly condensed) question is checked against the acronym glossary built at ingestion time (Section 2.4). This is a cheap, deterministic word-overlap lookup rather than a model call: each significant word in a glossary phrase is indexed against the acronym(s) it defines, and a question is only expanded with an acronym once it shares at least two significant words with that acronym's defining phrase — a single shared word is too weak a signal and would cause spurious expansions. When a match clears that bar, the acronym and its full phrase are appended to the retrieval query only, bridging the gap between a plain-language question (e.g. a question about a specific consultancy process) and the acronym-keyed vocabulary the documents actually use.

### 3.2 Why a Mapped Document? Primary Search and Secondary Search

Every question first goes through **topic matching** — comparing the question's meaning against a list of known policy topics (attendance, ragging, examinations, and so on). If the question is confident about one specific, known topic that has its own locally-indexed document, the system switches into **MAPPED_DOCUMENT** mode instead of searching the entire collection.

In practice this happens as two passes rather than one search that ranks a single document highest: PRIMARY searches the mapped document(s) alone, and — unless it already fills the full quota — a SECONDARY search then covers every other document afterward. Between the two, the whole collection is still reachable; the mapped document is simply searched first, and its chunks are always placed ahead of anything SECONDARY finds. (A topic doesn't always have a local document at all — some point only to an external link — and a topic that does can be backed by more than one document, in which case PRIMARY treats all of them together as one trusted scope.)

The reasoning: a college's attendance policy will use words like "leave," "absent," and "percentage" that also appear, in a different sense, in a completely unrelated financial-aid or hostel policy. Restricting the search to the document(s) the question is clearly about avoids pulling in loosely related chunks from other policies just because they happen to share common words.

Within MAPPED_DOCUMENT mode, there are two searches, run in a strict order of priority:

- **PRIMARY search** — searches only inside the matched document(s). Because this scope was already confirmed to be the right one, its chunks are treated as the most trustworthy source and are always placed ahead of anything else in the final answer.
- **SECONDARY search** — only runs if PRIMARY did not fill the full quota of chunks the system wants to send to the language model. It searches every other document (the matched document is deliberately excluded, not just the chunks already picked from it) to see if a second, genuinely relevant document exists — a question that legitimately spans two policies, for example.

This produces three distinct outcomes for PRIMARY, each handled differently:

- **EMPTY** — PRIMARY found nothing relevant at all. The system falls back to a full GLOBAL search (see below) across the whole collection, rather than trusting a topic match that turned out to be a false alarm.
- **PARTIAL** — PRIMARY found some, but not enough, relevant chunks. SECONDARY runs to fill only the remaining, unused slots — PRIMARY's own results are never displaced.
- **FULL** — PRIMARY alone already filled the quota (only possible when the overall chunk budget is at or below the PRIMARY cap). SECONDARY does not need to run at all.

If the question does not confidently match any known topic, the system uses **GLOBAL** mode instead: a single hybrid search across the entire policy collection, with no document restriction.

### 3.3 RRF and the Cross-Encoder

Both PRIMARY and GLOBAL searches are hybrid: they run a semantic (dense) search and a keyword (BM25) search side by side, then combine the two ranked lists using Reciprocal Rank Fusion (RRF). Dense search finds chunks with similar meaning even if the exact words differ; BM25 finds chunks that share exact words, acronyms, or policy codes that the embedding model might not weigh correctly. Combining both catches cases either one would miss on its own. How much each list counts toward the fused ranking is a configurable weight (equal by default) rather than fixed in code.

The fused list is then re-ordered a second time by a **cross-encoder** — a smaller, more accurate (but slower) model that reads the question and each candidate chunk together, rather than comparing two separately-computed vectors. Because it is slower, it is only run over a short list of already-promising candidates, not the entire collection.

**Two different relevance checks, for two different trust levels.** For GLOBAL and SECONDARY search, a chunk must clear the cross-encoder's own relevance score to survive — every candidate is judged the same way, whether it was found by dense search, BM25, or both.

PRIMARY search (the mapped document) is judged differently, on purpose. The document was already confirmed correct by the topic match, so its chunks are judged by the simpler dense similarity score instead of the cross-encoder score. This matters because the cross-encoder can genuinely score a short, heading-like chunk too low even when it is clearly the right answer — in testing, a chunk that was merely a section heading naming a staff welfare topic scored far below the cross-encoder's confidence threshold despite being the single best dense and keyword match inside its own document. Judging PRIMARY by the cross-encoder score would have thrown that chunk away and forced a worse, unnecessary fallback to GLOBAL search. The cross-encoder still gets to re-order PRIMARY's chunks — it just cannot drop one outright.

### 3.4 Relevance Filtering: A Two-Tier Floor, Not a Single Cutoff

The system is explicitly designed to say it doesn't know rather than answer confidently from weak or unrelated context — appropriate for a policy assistant, where a wrong answer about, say, an examination rule is worse than no answer. But a single hard cutoff turned out to be too blunt: a genuine near-miss (a chunk that is clearly the right topic but happens to score just under the confident threshold, often due to terse or unusually-worded policy phrasing) was being refused outright, identically to a question with no relevant content in the corpus at all.

Relevance filtering for GLOBAL and SECONDARY search is therefore a **two-tier floor**:

1. **Confident-match threshold.** If at least one candidate in the reranked pool clears this threshold, the normal path applies: every chunk that clears it is kept (up to the overall chunk budget), and the answer is presented with no special caveat.
2. **Low-confidence fallback floor.** If nothing clears the confident threshold, the system does not refuse immediately. Instead, it checks the single best-scoring candidate against a much lower floor. If that candidate clears the lower floor, it is kept and used anyway, but explicitly flagged as a low-confidence match: the user interface surfaces a visible warning telling the user to verify the answer against the source document directly, and the answer-generation model is separately instructed that it may use the flagged source if it genuinely supports the question, but must not add its own hedging language about the match being weak — the interface warning is the only place that caveat is shown, so it is not duplicated and does not read as an unsupported claim embedded in the answer text itself.
3. **Hard refusal.** Only when even the single best candidate falls below the low-confidence floor does the system refuse outright. This floor exists specifically to keep a genuinely off-topic question (one with no real relationship to anything in the policy corpus) from ever being dressed up as a low-confidence "answer" — those questions score dramatically lower than the kind of true near-miss the fallback tier is meant to rescue.

This tiering is deliberately asymmetric: the PRIMARY (mapped-document) relevance check described in Section 3.3 has no equivalent fallback. A mapped document with nothing above its own relevance threshold is treated as EMPTY and falls back to a full GLOBAL search, where it can still benefit from this two-tier logic on the whole collection — rather than being offered as a low-confidence match from a document that may simply be the wrong one.

### 3.5 Hyperparameters and Models

| Setting | Purpose |
|---|---|
| Final chunks to the language model | The overall budget of chunks sent to the language model per question. |
| PRIMARY allocation cap | Caps how many chunks a single mapped document can contribute, so a short, single-topic document cannot use up the entire budget and leave no room for a second, genuinely relevant document. |
| Topic match threshold | How confidently a question must match a known topic's name before MAPPED_DOCUMENT mode is used at all; below this, the system falls back to GLOBAL. |
| Confident-match reranker threshold (GLOBAL / SECONDARY) | The cross-encoder's own relevance score. A candidate at or above this is kept with full confidence. Chosen by scoring real question/chunk pairs and picking the value with the best precision/recall balance. |
| Low-confidence fallback floor | A much lower score than the confident threshold. The single best candidate below the confident threshold is still used, with a visible caveat, if it clears this floor; otherwise the system refuses. |
| Relevance threshold (PRIMARY only) | A dense cosine-similarity threshold used only for the mapped document's own chunks, instead of the reranker score — see Section 3.3 for why. Not the same scale as the reranker threshold above. |
| Hybrid candidates (GLOBAL mode) | How many results each of dense and BM25 search contributes before they are fused with RRF. |
| Dense / lexical fusion weight | How much each search method's ranking counts toward the fused RRF score. Equal by default — neither method is assumed to be more trustworthy without evidence. |
| Rerank candidate pool | How many fused candidates the cross-encoder scores, before filtering and the final chunk budget is applied. |
| Glossary expansion minimum word overlap | How many significant words a question must share with a glossary phrase before that acronym is appended to the retrieval query — guards against a single generic shared word triggering an irrelevant expansion. |

### 3.6 Justification for Major Design Decisions

- **Document routing before search, not after.** Deciding the search scope (one document vs. the whole collection) before running any expensive search avoids wasted work and, more importantly, avoids the risk of an unrelated document's chunks crowding out the right one in a single flat search. An earlier version of the system tried a softer approach instead — boosting a matched topic's chunks after a normal whole-collection search, rather than restricting the search itself. That was not strong enough: because reranking still re-scored the boosted pool from scratch, a strong topic match could still lose to an unrelated document. Restricting the search space up front removes that failure mode entirely instead of trying to out-tune it.
- **A small PRIMARY cap instead of letting PRIMARY use the whole budget.** Without this cap, a short policy document with several relevant-looking chunks could consume the entire chunk budget every time, permanently starving SECONDARY of any chance to surface a second relevant document — even on questions that genuinely need one.
- **Hybrid search over dense-only search.** Relying on the embedding model alone would occasionally miss an exact policy code, acronym, or proper noun that a keyword search catches easily. Running both and fusing the results is more robust than either alone.
- **A two-tier relevance floor instead of a single refuse/answer cutoff.** As described in Section 3.4, a single hard threshold conflated "genuinely off-topic" with "right topic, oddly worded" — both produced an identical refusal. Separating a confident-answer tier from a visibly-flagged low-confidence tier lets the system attempt a genuine near-miss instead of reflexively refusing, while still protecting against a confidently-wrong answer by making the uncertainty visible to the user rather than hidden.
- **Follow-up condensation and glossary expansion kept as two separate, narrow steps rather than one combined query-rewriting pass.** Each addresses a different, independent gap (missing context from conversation vs. missing institutional vocabulary), and keeping them separate means each can fail independently without affecting the other — a condensation failure still allows glossary expansion to run on the original question, and vice versa.
- **LangChain for generic search and model mechanics, custom code for institution-specific policy.** The embedding model, vector search, and cross-encoder reranker all run through LangChain's standard component interfaces rather than talking to each provider's SDK directly. This buys three things: the underlying model or vector store can be swapped by changing a configuration value instead of rewriting integration code; the team isn't maintaining bespoke, provider-specific client code for functionality (batching, retries, provider quirks) that a widely-used library already handles; and the same abstraction is reused in the generation layer (Section 4) to switch between a local and a hosted language model provider through one common interface. BM25 keyword search and the topic/document routing policy (MAPPED_DOCUMENT / PRIMARY / SECONDARY / GLOBAL) stayed custom on purpose instead of using an off-the-shelf LangChain retriever for them: that routing policy is institution-specific business logic with no generic equivalent to adopt, and, as the Section 3 introduction notes, off-the-shelf BM25 tooling cannot filter results by document after scoring — which the routing policy depends on — and lacks a safeguard against a false keyword "match" with no real word overlap in a small collection. Adopting LangChain where it fits a generic, well-solved problem and staying custom where the logic is genuinely specific to this system keeps the bespoke code limited to the parts that actually need to be bespoke.

### 3.7 Limitations / Improvements Needed

- **Thresholds are periodically re-tuned, not static.** Both the confident-match reranker threshold and the low-confidence fallback floor were chosen by scoring real question/chunk pairs rather than picked by hand, but the labelled set behind that tuning is still modest in size and should be revisited as more labelled questions accumulate. The topic-match threshold hasn't been tuned the same systematic way yet and is worth the same treatment.
- **The routing design assumes single-document answers by topic.** MAPPED_DOCUMENT mode deliberately treats a confident topic match as authoritative and scoped to that document (or document set); a question that genuinely needs deep information from several unrelated policies is not something the current one-shot SECONDARY search is built to fully cover.
- **Adjustable settings have been removed from the interface.** Earlier, the chunk budget and reranker threshold could be changed from the app's sidebar for a session, which risked a user unknowingly running with different settings than configured, with no record of what was actually used for a given answer. Those controls are gone now — every session runs on the same configured settings, visible per-answer only through the low-confidence warning described in Section 3.4.
- **Glossary expansion is purely lexical.** It cannot catch a jargon gap that doesn't involve a recognizable acronym, or a paraphrase that shares no significant words with the glossary phrase at all.

## 4. Response Generator Design

Once retrieval has produced a short, ranked list of relevant chunks, the Response Generator builds the final answer. This stage has one strict rule: the language model must answer only from the chunks it was given, and must cite exactly which chunk (and therefore which document and page) each part of its answer came from.

The **Context Builder** assembles the retrieved chunks into a numbered list with their source and page clearly marked, so the model has something concrete to cite. The **language model** then generates an answer using only that numbered context, plus a capped window of recent conversation turns so it can resolve references like "it" in its reply. Finally, the **Response Processor** checks that every citation the model used actually exists in the numbered context — a citation to a chunk that was never provided is treated as an error, not trusted — and formats the sources shown to the user. If the pipeline ends up refusing to answer (no grounded content found), a separate, best-effort model call suggests an alternative phrasing of the question the user could try instead.

### 4.1 Model and Parameters

| Setting | Value | Purpose |
|---|---|---|
| Provider | Configurable: a local, self-hosted model runtime, or a faster hosted inference provider | The local option keeps every question and every policy chunk fully on the local machine; the hosted option trades that guarantee for faster inference. Only the selected provider needs to be reachable. |
| Local model | A small instruction-tuned model | Small enough to run acceptably on CPU-only hardware. |
| Hosted model | A larger hosted model | Used when local CPU inference is too slow for the deployment. |
| Temperature | 0 (default) | The model is asked to be as consistent and literal as possible by default, not creative — appropriate for a policy lookup tool where the same question should get the same answer. |
| Answer length cap (local) | A few hundred tokens | Keeps worst-case answer time predictable on CPU-only inference, and reflects that a policy answer never legitimately needs to be very long. |
| Answer length cap (hosted) | Higher than the local cap | The hosted provider isn't CPU-latency constrained, so this is a plain answer-length ceiling rather than a latency safety valve. |
| Conversation history sent to the generation model | A short, fixed window of the most recent messages | Keeps the prompt short and generation fast rather than growing unbounded over a long session. This is deliberately narrower than what is persisted for the session as a whole — see Section 5. |

### 4.2 Justification for Major Design Decisions

- **Answer-only-from-context instruction, enforced twice.** The model is told to only use the provided chunks, and the Response Processor separately checks every citation against what was actually retrieved. Relying on the model's instruction-following alone would not be safe enough for a policy assistant — the second, code-level check catches the cases where the model doesn't follow the instruction perfectly.
- **Temperature 0 by default.** For a factual lookup tool, consistency between repeated runs of the same question matters more than varied or creative phrasing. It is fixed by configuration rather than adjustable in the app, so the default is what the assistant answers with unless the configuration itself changes.
- **A hard token cap on local generation.** Without a cap, a slow CPU-only model could keep generating well past a reasonable wait time; a lower cap trades a small risk of a truncated answer for a predictable, bounded worst-case response time.
- **Provider is a configuration switch, not a code change.** Every part of the pipeline above the language-model call only ever talks to LangChain's provider-agnostic chat model interface (the same abstraction used for embedding and reranking in Section 3), so switching between a fully offline and a hosted deployment is a one-line configuration change rather than a different code path per provider.
- **A refusal is followed by a rephrasing suggestion, not left as a dead end.** A plain "nothing found" refusal is the one outcome where the user has no obvious next step; generating one alternative phrasing (as a distinct, best-effort model call that degrades silently on failure) gives them something concrete to try instead of just a closed door. This suggestion is deliberately not offered for the topic-matched-but-unindexed refusal case, since that case already hands the user a concrete official link instead.
- **The low-confidence caveat lives only in the interface, never in the generated answer text.** As described in Section 3.4, the model is explicitly told not to add its own hedging language about a flagged low-confidence source. Keeping the warning in exactly one place avoids it being duplicated, and keeps the generated answer text itself a clean, evaluable claim grounded in the given context — an answer that editorialized about its own confidence would be harder to judge for faithfulness to the retrieved content.

### 4.3 Limitations / Improvements Needed

- **No streaming to the user.** The full answer is generated before anything is shown, which can feel slow on CPU-only local inference; a token-by-token streaming response would feel more responsive.
- **Conversation memory available to generation is narrow.** The capped window of recent messages sent to the generation model is intentionally small to bound prompt size, which limits how far back a genuine multi-turn conversation can be resolved purely from what the model itself sees — follow-up questions are instead primarily handled on the retrieval side (Section 3.1), not by giving the generation model more history.
- **A truncated answer is not obviously flagged to the user.** If the local provider's token cap cuts an answer short, the user currently has no visible signal that the answer was cut off rather than complete.

## 5. Session and Conversation Infrastructure

Every question is asked within a session, identified by the asking user's email address rather than anonymously, so conversation history can be kept per person and multiple concurrent users each get their own conversation thread. A session that has been inactive past a configured timeout is treated as expired, and a fresh session begins for that user — with a countdown warning shown before the cutover so an idle user isn't surprised by a reset mid-thought.

Session and message history are persisted durably, independent of the capped, short conversation window that is actually sent to the generation model for any single answer (Section 4.1). This separation is deliberate: the full history is what a user sees as their chat transcript and what per-answer timing statistics are computed over, while only a narrow recent slice of it is ever fed back into a model call, keeping prompt size and latency predictable regardless of how long a conversation has run. Each stored assistant message also carries how long that turn took to produce, split into its retrieval and generation portions, which is surfaced in the interface as a rolling average response time computed over only the most recent answers — so it reflects current system performance rather than being dragged by the session's entire history.

### 5.1 Repeat-Question Caching

A separate, optional caching layer sits in front of retrieval to avoid redundant work on questions that are asked often, in essentially the same words, by many different users — a common pattern for a small set of frequently-asked policy questions. This cache is deliberately narrow in scope:

- It matches only on the normalized, exact text of a question — no fuzzy or semantic matching — so a cache hit is always provably the same question asked before, never an approximate match that could silently serve the wrong retrieved chunks for a similar-but-different query.
- A question only becomes eligible for caching once it has been asked some minimum number of times; a first-time question always goes through full retrieval, so the cache only ever holds genuine repeat traffic rather than being populated speculatively.
- It is wiped entirely whenever ingestion completes, since cached chunks are tied to the exact contents of the index they were retrieved from, and a re-ingestion can add, remove, or renumber chunks in ways that would make a stale cache entry serve content that no longer matches the live index.
- Any failure of the caching layer itself (the cache being unreachable, timing out, etc.) falls back transparently to calling retrieval directly — it is a performance optimization, never a hard dependency for answering a question.
- Evaluation and experimentation runs that override retrieval settings for a single call bypass the cache entirely, so an experimental run never pollutes the cache with results computed under non-default settings, and never silently receives a cached result from the app's normal configuration instead of the override it asked for.

### 5.2 Justification for Major Design Decisions

- **Session identity tied to an email address, not anonymous.** This lets conversation history be traced back to a person, which matters for a small, known user population (college staff/students) more than it would for an open public-facing tool, and enables the per-user conversation continuity the interface relies on.
- **A short inactivity timeout with a visible warning.** A policy-lookup session is typically a few quick questions in a row, not an all-day open conversation; expiring an idle session keeps stored history meaningfully scoped to one sitting, while the warning avoids silently discarding an in-progress thought.
- **Durable full history decoupled from the model's working context window.** Persisting everything while only feeding a narrow slice to the model gets the best of both: a complete, inspectable transcript for the user and for diagnostics, without the prompt-size and latency cost of replaying an entire session to the model on every turn.
- **Exact-match-only caching instead of a semantic cache.** A semantic/fuzzy cache key risks serving one question's cached chunks for a different, merely similar-sounding question — unacceptable for a policy assistant where precision matters more than the marginal performance gain from a looser cache. Restricting to exact matches trades some cache coverage for a strict correctness guarantee.

### 5.3 Limitations / Improvements Needed

- **The repeat-question cache is global, not per-topic-aware.** It caches on raw question text only; two differently-worded questions that are retrieval-equivalent still each have to independently clear the repeat-count threshold before either is cached.
- **No cross-session continuity.** A user who returns after their session has expired starts with no memory of the prior conversation, even if it happened minutes after the timeout.

## 6. Test Process and Experimentation

The system is checked in two independent ways: whether retrieval found the right chunks, and whether the generated answer is actually good. Keeping these separate matters — a bad answer could be a retrieval problem or a generation problem, and conflating the two into one score makes it hard to tell which stage needs fixing.

### 6.1 IR Metrics (Retrieval-only, No Language Model Involved)

IR stands for Information Retrieval — the long-established field concerned with finding relevant documents for a query, going back to library and web search long before language models existed. These metrics only ask: out of everything retrieval returned, was it the right material? They involve no language model call, so they are fast and free to run, and are used as the primary tool while iterating on retrieval settings.

### 6.2 LLM-Judged Evaluation (End-to-End)

A second evaluation suite scores the answers the live pipeline actually produces, using a separate language model as a judge — deliberately a different model from the one that answers real questions, so the judge is not grading its own work. See the Appendix for how an LLM judge works in more detail. Judged qualities include whether every claim in the answer is actually backed by the retrieved chunks (faithfulness), whether the answer actually addresses the question (relevancy), and how well it matches a known-correct reference answer, alongside retrieval-quality metrics scored over the same live-pipeline run.

Because this suite calls an external judge model for every test question, it is slower and more expensive than the IR metrics above. In practice, IR metrics are used as a fast, free first check while iterating on retrieval settings, and the LLM-judged suite is reserved for confirming that the final, best-looking configuration also produces good end-to-end answers.

Every evaluation run can be compared against a previously saved snapshot of results, so a regression or improvement in a specific metric is visible as a delta rather than requiring a manual before/after comparison.

### 6.3 Tracked Experimentation

Beyond pass/fail evaluation, retrieval hyperparameters (chunk budget, fusion weights, thresholds, candidate pool sizes, and similar settings) are tuned through a tracked experimentation practice rather than ad hoc trial and error: a given configuration is run against the IR metrics suite and logged as one comparable, timestamped experiment run, so different settings can be compared against each other after the fact rather than only against whatever was tried most recently. An experiment that changes something about how the index itself is built (the embedding model, or chunking settings) is run against a separately-built, dedicated index rather than the index the live application actually serves, so experimentation never risks disturbing the deployed system; a matching index is built once and reused for later experiments with the same settings.

This tracked-experimentation practice is retrieval-only — it never invokes the LLM-judged evaluation suite, so trying a new hyperparameter value never costs a judge-model call.

### 6.4 Justification for Major Design Decisions

- **Two independent test suites instead of one blended score.** A single end-to-end "quality" number cannot distinguish a retrieval failure from a generation failure, which is precisely the distinction needed to know what to fix. Keeping IR metrics and LLM-judged evaluation separate preserves that distinction.
- **A separate judge model from the answer model.** A model judging its own output is a weaker signal than an independent one — a shared blind spot or stylistic bias could inflate scores without the model ever actually improving.
- **Retrieval-only tracked experimentation, kept apart from the LLM-judged suite.** Most hyperparameter tuning only affects which chunks are found, not how the answer is phrased — resolving that with the free, fast IR metrics first and reserving the expensive judged suite for confirming a final candidate configuration keeps iteration cheap without sacrificing a final end-to-end check.
- **Baseline comparison built into the evaluation workflow.** Surfacing a metric's delta against a saved baseline automatically, rather than requiring someone to manually diff two result sets, makes a regression far more likely to be caught before it ships.

### 6.5 Limitations / Improvements Needed

- **The labelled test set remains modest in size.** Retrieval and generation thresholds are tuned against real question/chunk pairs rather than picked by hand, which is a real improvement over guessing, but the underlying labelled set is still small enough that conclusions drawn from it should be revisited as more labelled questions are added.
- **The test data doesn't fully agree with the mapped-document design yet.** Some existing labelled test questions were written assuming an answer could legitimately draw on more than one document, but MAPPED_DOCUMENT mode deliberately treats a confident topic match as authoritative and single-document-scoped. This is a mismatch between what the test expects and what the system is designed to do, not a bug in either one — it needs a decision on which is right, and the test data or the design should be reconciled accordingly.
- **Multi-document coverage is under-tested.** The current labelled test set does not include many questions that genuinely need two or more unrelated policies, so the SECONDARY/GLOBAL cross-document behavior has limited automated coverage against that specific case.

## Appendix: Key Concepts

Short, plain-language explanations of the core techniques used in this system, for anyone encountering them for the first time.

### A.1 BM25

**What it is:** A keyword-matching search method. It scores how well a chunk matches a query based on which exact words they share, giving more weight to rare, distinctive words than to common ones.

**Why it's needed here:** Semantic (meaning-based) search can miss an exact policy code, acronym, or proper noun if the embedding model doesn't represent it well. BM25 catches literal word matches that semantic search sometimes misses, so running both together is more reliable than either alone.

### A.2 Dense Search

**What it is:** Search based on meaning rather than exact words. Every chunk and every query is turned into a numeric vector (an embedding); chunks whose vector is close to the query's vector are considered a good match, even if they don't share any exact words.

**Why it's needed here:** It lets the system find a relevant chunk even when the question is phrased completely differently from the policy text — for example, matching "how many classes can I miss" to a policy chunk that says "minimum attendance requirement," without either phrase sharing a single word.

### A.3 RRF (Reciprocal Rank Fusion)

**What it is:** A simple way to combine two separately-ranked result lists (here, dense search's list and BM25's list) into one list, based on each chunk's rank position in each list rather than its raw score. A chunk that ranks highly in both lists ends up ranked highly in the combined list.

**Why it's needed here:** Dense search and BM25 produce scores on completely different, incompatible scales — a dense similarity score and a BM25 keyword score cannot be directly added together. RRF sidesteps that problem by comparing rank positions instead of raw scores.

### A.4 Cross-Encoder Reranker

**What it is:** A model that reads the question and one candidate chunk together, at the same time, and outputs a single relevance score for that specific pair. This is more accurate than comparing two separately-computed embeddings, but much slower, so it can only be run over a short list of candidates, not the whole collection.

**Why it's needed here:** It acts as a final, more careful quality check after the faster dense+BM25 search has narrowed the field down to a manageable shortlist — catching ordering mistakes the faster methods make.

### A.5 Relevance Filtering

**What it is:** A minimum-score cutoff applied after reranking. A candidate chunk scoring below the cutoff is either dropped entirely or, for the single closest candidate only, kept with a visible low-confidence caveat rather than presented as a normal result — see Section 3.4 for the full two-tier policy used in this system.

**Why it's needed here:** It is what allows the assistant to say "I don't know," or to clearly flag genuine uncertainty, instead of silently guessing from a weak or unrelated chunk — a deliberate safety choice for a policy assistant, where a confident wrong answer is worse than no answer.

### A.6 Models Used in This System

- **Embedding model (BAAI/bge-base-en-v1.5):** A mid-sized, CPU-friendly sentence-embedding model that turns both chunks and queries into vectors for dense/semantic search; chosen for stronger paraphrase matching than a smaller, faster alternative.
- **Reranker model (BAAI/bge-reranker-v2-m3):** A cross-encoder model that scores each (question, chunk) pair jointly for the final, most accurate relevance pass before chunks are sent to the answer-generation model.
- **Local answer model (Llama 3.1 8B, served via Ollama):** A small instruction-tuned model used to generate answers when the deployment runs fully offline on CPU-only hardware.
- **Hosted answer model (Qwen3 8x27B, served via Groq):** A larger, hosted model used to generate answers when faster inference is preferred over a fully local, CPU-bound deployment.
- **Judge model (Gemini 3.1 Flash Lite):** An independent, hosted model used only during evaluation to score the pipeline's own answers — never used to generate answers for real users, so it isn't grading its own work.

### A.7 How an LLM Judge Works

**What it is:** An automated evaluation approach that asks a separate large language model to read a question, the answer the system produced, the chunks that were retrieved, and (where available) a known-correct reference answer — and then to score specific qualities of that answer, such as whether every claim is actually backed by the retrieved chunks (faithfulness) or whether the answer actually addresses the question (answer relevancy). The judge is deliberately a different model from the one that generated the answer, so it is not grading its own work.

**Why it's needed here:** Some qualities of a good answer — like whether it is genuinely grounded in the source material, or genuinely on-topic — are very hard to check with a simple formula. A language model reading the answer critically can catch nuanced problems (a subtly unsupported claim, a technically-relevant-but-unhelpful answer) that a purely numeric metric like Recall or Precision cannot.
