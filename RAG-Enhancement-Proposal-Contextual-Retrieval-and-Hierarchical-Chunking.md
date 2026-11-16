# Proposed Enhancement: Contextual Retrieval, Hierarchical Chunking, and Learned Reranking

This document proposes three related changes to the ingestion and retrieval stages of the CMRIT College Policy RAG Assistant: **contextual retrieval** (enriching each chunk with document context before it is embedded), **hierarchical (parent-child) chunking** (retrieving on small, precise chunks but generating from their larger surrounding section), and **learned reranking** (replacing or supplementing the cross-encoder's relevance judgment with a model trained on this system's own retrieval signals). The first two target the same underlying weakness — fixed-size chunking throws away the surrounding context a short passage needs to be matched and judged correctly — and are designed to be adopted together, though each could be adopted independently. The third is a separate, lower-priority optimization that improves how multiple existing relevance signals are combined, rather than changing what is indexed.

## 1. Motivation

The current design already surfaces two symptoms of this weakness directly:

- A short, heading-like chunk can score far below its true relevance under the cross-encoder reranker, because the reranker has only the bare chunk text to judge — no indication of which policy, section, or surrounding rule it belongs to. The system currently works around this for mapped-document search by trusting dense similarity instead of the reranker score for that one search path, but the underlying problem (a chunk that reads as context-free) is not actually fixed, and the same weakness is unmitigated everywhere else (SECONDARY and GLOBAL search).
- Several documents share long stretches of identical boilerplate, which was addressed by dropping duplicate chunks outright at ingestion. That fix treats the symptom (duplicate text competing for the same retrieval slot) rather than the cause (a chunk with no context of its own to distinguish it from a superficially similar one elsewhere).

Both proposed changes attack the cause directly: give each indexed unit enough of its own context to be matched and judged on its actual meaning, independent of which other chunks happen to look similar.

## 2. Contextual Retrieval

### 2.1 What It Is

Before a chunk is embedded, a short, chunk-specific summary of where it sits in its source document is generated and prepended to the chunk's text for the purposes of embedding and keyword indexing only — for example, a bare passage about casual-leave eligibility would be indexed alongside a prepended note identifying it as being from the staff leave policy's eligibility section, even though that identifying detail never appeared in the passage itself. The chunk's original, unmodified text is still what is shown to the user and sent to the answer-generation model; the added context exists purely to make the chunk easier to find and to judge correctly during retrieval.

### 2.2 How It Would Fit Into the Pipeline

This becomes a new ingestion stage, positioned after chunking and deduplication but before embedding:

1. Chunking and deduplication run exactly as they do today, so this stage never contextualizes a chunk that will be dropped as a duplicate.
2. For each surviving chunk, a language model is given the chunk's text together with enough of its surrounding document (or at minimum its section heading and document title) to generate a short, factual description of what the chunk is and where it belongs — not a paraphrase of the chunk's content, just its place in the document.
3. That description is prepended to the chunk's text to form the string that is actually embedded and indexed in BM25. The chunk's stored, citable text — what the Context Builder assembles for the answer-generation model and what the Response Processor maps to a source reference — remains the original, uncontextualized text.

Dense search, BM25 search, and the cross-encoder reranker all then operate on context-enriched text without any change to how they are called; this is purely an ingestion-time change to what gets indexed, not a change to the retrieval or generation stages' logic.

### 2.3 Hyperparameters and Design Considerations

| Setting | Consideration |
|---|---|
| Context-generation model | A smaller, cheaper model than the answer-generation model is appropriate here — the task is narrow (describe a passage's place in its document) and runs once per chunk at ingestion time, not on the live query path, so neither latency nor top-tier reasoning quality is the binding constraint. |
| Context length | Kept deliberately short (roughly a sentence) — the goal is to disambiguate the chunk, not duplicate its content; a long generated context risks the embedding representing the context more than the chunk itself. |
| Surrounding-document window given to the model | Needs enough of the source document around the chunk (preceding heading, surrounding paragraphs, document title) for the model to describe the chunk's place accurately, without passing the entire document to every chunk's context call. |
| Caching / incremental regeneration | Since ingestion already rebuilds the whole index from scratch (Section 2.6 of the main design document notes this as an existing limitation), contextualization inherits the same cost profile — every chunk gets a fresh context call on every ingestion run. This is the main new resource cost this change introduces. |

### 2.4 Justification

- **Fixes the cause, not the symptom, of the heading-like-chunk problem.** Rather than special-casing which search path trusts the reranker (as the current design does for mapped-document search), every chunk becomes self-describing enough to be judged on its own meaning by any search path, including the ones with no such workaround today.
- **Leaves generation and citation entirely unaffected.** Because the generated context is only ever used for embedding/indexing and never shown to the user or the answer-generation model, this change carries none of the risk of the model treating a generated description as retrieved fact — it cannot leak into an answer or a citation.
- **A one-time ingestion cost, not a live-query cost.** Every architectural decision elsewhere in this system treats query-time latency as precious (the CPU-only local generation cap, the narrow conversation-history window, the repeat-question cache); this change deliberately spends cost at ingestion time, where it is far cheaper to absorb, instead.

### 2.5 Limitations / Risks

- **Ingestion time and cost grow with the number of chunks.** A full re-ingestion now requires one additional model call per chunk, on top of embedding; for a larger document set this would make the existing "no incremental ingestion" limitation considerably more expensive to live with, and incremental ingestion becomes a more pressing need than it is today.
- **The generated context can itself be wrong.** A context-generation call can mischaracterize a chunk's place in the document (attributing it to the wrong section, for example), which would then mislead retrieval the same way bad metadata would — this needs spot-checking after the first full re-ingestion, not blind trust.
- **It does not replace the existing PRIMARY dense-score trust workaround outright.** That workaround should likely remain in place as a second layer of defense even after this change, at least until there is confidence that contextualized embeddings reliably fix the heading-chunk problem on their own.

## 3. Hierarchical (Parent-Child) Chunking

### 3.1 What It Is

Rather than indexing and generating from the same unit of text, retrieval and generation use two different granularities: small "child" chunks (closer to the current chunk size) are what dense search, BM25, and reranking actually match and score against the query, while each child is linked to a larger "parent" unit — typically a whole section or clause — that is what actually gets sent to the answer-generation model once a child chunk is selected.

### 3.2 How It Would Fit Into the Pipeline

1. During chunking, each page or section is first split into parent units (larger, coherent sections — a clause, a numbered policy item, a subsection), and each parent is then further split into the smaller child chunks used today, with each child recording which parent it belongs to.
2. Child chunks are what get embedded, indexed in BM25, and (optionally) contextualized as described in Section 2 — contextual retrieval and hierarchical chunking compose naturally, since the context-generation step already needs to look at a chunk's surrounding document, which a parent unit conveniently provides.
3. Retrieval, routing, and reranking all continue to operate on child chunks exactly as they do today — topic matching, MAPPED_DOCUMENT routing, hybrid search, fusion, and the cross-encoder reranker are all unaffected, since they only need the smaller, precise unit to score accurately against the query.
4. Only at the point where a child chunk is about to be included in the final chunk budget does its parent get substituted in: the parent's full text becomes what the Context Builder assembles into the numbered context for the answer-generation model, while the citation shown to the user still carries the child's precise page/section location, not the whole parent's span.

### 3.3 Hyperparameters and Design Considerations

| Setting | Consideration |
|---|---|
| Parent unit size | Should align with a natural document boundary (a numbered clause, a subsection) rather than a fixed character count, so a parent is a coherent, self-contained unit of policy rather than an arbitrary window. |
| Child chunk size | Can likely stay close to the existing chunk size, since its only job is precise matching, not providing enough context to answer from directly. |
| Final chunk budget | Needs re-evaluation once parents (not children) are what actually reach the answer-generation model — a parent is larger than a child, so the same token budget may need a smaller chunk count to avoid overflowing the model's context window or the configured answer-length cap. |
| Deduplication's interaction with this | Deduplication (Section 2.2 of the main design document) currently runs on whole-chunk text; it would need to run at the child level, since that's still the unit that can be duplicated near-verbatim across documents, while parent de-duplication is a separate, coarser question this change does not attempt to solve. |

### 3.4 Justification

- **Separates the concern of matching precisely from the concern of having enough context to answer from.** Today, one chunk size has to serve both jobs at once — small enough to be a precise match, large enough to carry enough context to answer from — which is an unavoidable compromise. Splitting the two lets each be sized for what it's actually for.
- **Directly extends what contextual retrieval already provides.** A context-generation call already needs to see a chunk's surrounding document to describe its place in it; a parent-chunking structure gives that lookup a natural, already-computed boundary to use, rather than needing a separate ad hoc window.
- **Keeps citation precision unchanged.** Because the citation shown to the user is still derived from the child chunk's own page/section, not the parent's broader span, this does not weaken the existing guarantee that a citation points at a specific, narrow location rather than a vague "somewhere in this section."

### 3.5 Limitations / Risks

- **Larger context per chunk sent to generation.** Since a parent is larger than today's chunk, the same number of chunks in the final budget now costs more tokens — this will likely require lowering the chunk-budget setting, trading breadth (number of distinct sources considered) for depth (more complete context per source), and that trade-off should be tuned deliberately rather than left at today's values.
- **Two units of text to keep consistent.** Ingestion now has to maintain a parent-child relationship correctly through deduplication, re-ingestion, and metadata assembly; a bug here (a child pointing at a stale or missing parent) is a new failure mode this system does not have today.
- **The PRIMARY allocation cap and relevance-filtering logic (Section 3 of the main design document) operate on the same count of items whether they're child chunks or today's flat chunks, but that count now represents a coarser slice of the document once parents are substituted in** — the existing caps may need retuning rather than assumed to still represent the same effective amount of content.

## 4. Learned Reranking

### 4.1 What It Is

Rather than relying solely on the cross-encoder's score (or, for PRIMARY search, dense similarity) as the single signal that decides relevance, a small, fast machine-learned ranking model — a gradient-boosted tree model such as XGBoost's ranking objective is a natural fit — is trained to combine several retrieval signals this system already computes into one learned relevance judgment, rather than relying on a single hand-picked signal and a single hand-tuned threshold.

### 4.2 How It Would Fit Into the Pipeline

This is a retrieval-stage change only, with no effect on ingestion, chunking, or generation:

1. For each fused candidate, assemble a feature vector from signals already available at that point in retrieval: the dense similarity score, the BM25/lexical score, the RRF-fused rank, the cross-encoder score, the topic-match similarity (when in MAPPED_DOCUMENT mode), whether the candidate was found by dense search, lexical search, or both, and simple chunk-level features such as text length.
2. A ranking model, trained offline on the labelled question/chunk pairs already used to tune today's thresholds, scores each candidate from these features.
3. That learned score replaces (or is blended with) the cross-encoder score as the basis for the confident-match / low-confidence / refuse tiering described in the main design document's relevance-filtering section — the tiering logic itself does not need to change, only what produces the score it tiers on.
4. The cross-encoder itself is not necessarily removed — its score becomes one input feature among several, rather than the sole relevance judgment, so this change can be adopted incrementally without discarding the existing reranking step.

### 4.3 Hyperparameters and Design Considerations

| Setting | Consideration |
|---|---|
| Training data | The same labelled question/chunk pairs already used to tune the confident-match and low-confidence thresholds — no new labeling effort is strictly required to get started, but the model's quality is bounded by how much of that labelled set exists. |
| Feature set | Limited to signals already computed during retrieval, so scoring a candidate at query time costs a cheap feature lookup and a small model inference, not a new expensive computation. |
| Model choice | A gradient-boosted tree ranker over a neural ranking model — it trains quickly and reliably on small tabular datasets, is fast enough at inference time to run on every candidate without a noticeable latency cost, and its feature importances are inspectable, which matters for a system whose design already values explainability over opacity. |
| Retraining cadence | Should be retrained whenever the labelled set grows meaningfully, or whenever an upstream signal changes (a new embedding or reranker model, for instance) enough that previously-learned feature weightings may no longer be well calibrated. |

### 4.4 Justification

- **Combines signals the system already computes but currently uses separately or in isolation.** Dense score, lexical score, RRF rank, and cross-encoder score each carry partial information about relevance today; a learned combination can weigh them against each other more effectively than any one of them used alone, or than a fixed linear fusion weight.
- **Cheap at query time.** Because every input feature is already produced somewhere in the existing retrieval path, scoring with a small tree-based model adds negligible latency compared to the cross-encoder pass that already runs.
- **Reuses existing evaluation infrastructure.** The same labelled data that justified today's thresholds is what trains and validates this model — no new data pipeline or labeling effort is required to try it.

### 4.5 Limitations / Risks

- **Bounded by the size of the labelled set.** This is the binding constraint, not the modeling technique: a ranking model trained on a small labelled set risks fitting quirks specific to those questions rather than learning a relevance judgment that generalizes to new ones, and may not outperform the pretrained cross-encoder's score on its own, which required no labels from this system at all to work reasonably well.
- **Adds a model-maintenance responsibility that doesn't exist today.** Unlike the pretrained embedding, reranker, and language models — which are adopted as-is and simply swapped by configuration — this model is trained on this system's own data and needs to be retrained, versioned, and re-evaluated as that data and the upstream signals it depends on change over time.
- **Lower priority than the other two enhancements in this document.** It optimizes how existing signals are combined; it does not address a case those signals currently have no information about at all, the way contextual retrieval and hierarchical chunking do for a context-free, heading-like chunk. It is best adopted after the labelled set has grown and after the first two enhancements have already changed what those underlying signals look like.

## 5. How the Three Enhancements Combine

Contextual retrieval and hierarchical chunking change what is indexed and what is handed to generation, while learned reranking changes how the resulting candidates are scored and ordered — the three are complementary rather than competing. Adopted together, the smallest indexed and matched unit is a context-enriched child chunk; a learned ranking model scores each candidate using the dense, lexical, cross-encoder, and topic-match signals computed against that enriched unit; and once a child chunk is selected, its larger parent section is what is actually handed to the answer-generation model. This order of adoption also matters for sequencing: because learned reranking is trained on signals computed against whatever is currently indexed, it is most valuable adopted last, after contextual retrieval and hierarchical chunking have already changed what those underlying signals look like — training it against today's signals first would mean retraining it again once the other two changes land anyway.

## 6. Evaluation Plan

None of these changes should be adopted purely on the strength of this reasoning — all three should go through the same evaluation discipline the system already uses for retrieval changes:

1. Build a separate, dedicated index under the contextualized/hierarchical scheme, leaving the live application's index untouched, consistent with how retrieval experiments are already isolated from the deployed index today.
2. Run the existing retrieval-only IR metrics against that index first, as a fast, free check, before considering the more expensive LLM-judged evaluation suite.
3. Specifically check whether previously-identified weak cases (a heading-like chunk inside its own mapped document, boilerplate-adjacent content) rank differently than they do today, since those are the concrete failures this proposal is meant to fix.
4. Only after IR metrics show improvement, run the full LLM-judged evaluation suite to confirm end-to-end answer quality — including faithfulness, since generation now draws on a larger parent passage per source and should be checked that it doesn't start incorporating irrelevant material from elsewhere in that larger passage.
5. If learned reranking is adopted, evaluate it with the same held-out discipline already used elsewhere — trained only on a labelled training split, scored against IR metrics on a held-out split it never saw during training, so its apparent improvement isn't simply it having memorized the training questions.
6. Compare the chosen final configuration against the current system's saved baseline on both suites before treating any of the three changes as adopted, not just experimented with.

## 7. Open Questions

- Whether context generation should run against the same local/hosted provider split already used for answers, or default to whichever is cheaper at the chunk volumes this corpus implies.
- How parent boundaries should be detected for documents that aren't cleanly structured into numbered clauses or consistent headings.
- Whether the chunk budget and PRIMARY allocation cap need to change in absolute terms, or whether they turn out to scale naturally once each slot represents more content.
- Whether incremental ingestion (already an open limitation today) becomes necessary rather than merely desirable once contextualization adds a per-chunk model call to every full re-ingestion.
- How large the labelled set needs to grow before a learned reranker can be expected to outperform the pretrained cross-encoder's score on its own, and what the lowest-effort way to grow it is (e.g. promoting real low-confidence or refused queries, once reviewed, into the labelled set).
