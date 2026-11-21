# CLAUDE.md

## Known issue: wrong-chunk answers on dense numbered-note policy pages

**Symptom:** For "how much can be claimed for extra papers published with students," the
chatbot answers with only a bare number (e.g. "Rs. 8,000/- per paper") instead of the full
eligibility/limit/process detail, and the number can come from the *wrong* note entirely.

**Root cause:** `RCPIE Targets.pdf` page 2 is one dense, unstructured block of text covering
notes 9–21 back-to-back with no headings. Chunking is a fixed 800-char / 120-char-overlap
window (`config/config.yaml` → `chunking.chunk_size` / `chunk_overlap`), with no awareness of
note boundaries. Two unrelated notes on that page both mention "Rs. 8,000/-" (note 15:
conference registration fee reimbursement; note 18: extra faculty papers with students). The
fixed-size window can split a note's full text across chunk boundaries and produce a chunk
that scores well on shared vocabulary ("Rs. 8,000", "paper", "reimbursement") for the wrong
note, while the chunk that actually contains the asked-about note scores too low to make the
top-k.

**Masking problem in the test suite:** `tests/_07_evaluation/test_retrieval_functional.py`'s
`recall_at_k` matches at page granularity (`source_file` + `page_number`), so a case like this
can report a "hit" (page 2 was retrieved) even though the specific chunk retrieved is the wrong
note. The functional test passes while the real answer is still wrong.

**Fix approach (not yet applied):**
1. Chunk dense numbered-note pages on note boundaries (e.g. split on `\d+\.\s`) instead of a
   blind fixed-size window, so each note becomes one self-contained, individually-retrievable
   chunk — note 15 and note 18 stop competing for the same chunk space.
2. Tighten the relevant test case(s) in `tests/retrieval-functional-cases.json` to use the
   `text_contains` field already supported by the test harness (see docstring in
   `test_retrieval_functional.py`), e.g.:
   ```json
   "must_have": [{"source_file": "RCPIE Targets.pdf", "page_number": 2, "text_contains": "extra Faculty papers with students"}]
   ```
   so page-level recall can no longer mask a wrong-note hit.
3. Before assuming retrieval is missing content, check whether it's actually missing from the
   source doc. The "RCPIE FAQ.docx Q41" conflicting Rs. 3,000 answer referenced as an expected
   answer for this question does not appear anywhere in the currently ingested
   `RCPIE FAQ.docx` (checked all paragraphs and all 44 tables) — it may be from a stale/older
   FAQ version not in `knowledge_documents/`, not a retrieval gap.

**Diagnostic approach used to find this** (repeatable for similar one-liner/wrong-answer
reports): run the retrieval service directly for the exact question (mirrors the fixture setup
in `test_retrieval_functional.py`) and print the actual top-k chunks' `source_file`,
`page_number`, and text — don't rely on recall/precision numbers alone, since page-level
metrics can hide a wrong-chunk-on-the-right-page failure.
