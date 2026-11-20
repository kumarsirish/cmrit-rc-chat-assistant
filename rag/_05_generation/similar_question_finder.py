"""Finds a real, previously-asked question nearly identical to one the
pipeline just failed to answer.

A stronger "did you mean" suggestion than rephrase_suggester's
LLM-generated guess (see that module): a question returned here is one
we already know got a real, grounded answer — not just a plausible-
sounding rewrite the LLM invented on the spot. Used as the first thing
tried on a "nothing found" refusal (see rag_pipeline.RagPipeline._finalize),
falling back to rephrase_suggester only when nothing here clears the
similarity threshold.

Matched by embedding cosine similarity (the same sentence-transformers
model already loaded for retrieval — see
rag._01_ingestion.embeddings.EmbeddingGenerator), computed fresh on
every refusal rather than cached: this only ever runs on the cold,
rare "nothing found" path, not the hot retrieval path, so there's no
persistent matcher to keep in sync with new questions the way
topic_matcher.py's is for documents.
"""

import math


def _cosine_similarity(vector_a: list, vector_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_similar_question(
    question: str, candidate_questions: list, embedding_generator, similarity_threshold: float = 0.90
) -> str:
    """Returns the candidate_questions entry most similar to question,
    if its cosine similarity clears similarity_threshold — None if
    none do, candidate_questions is empty, or the only match is
    question itself (asking the user "did you mean the exact same
    question?" would be useless)."""
    distinct_candidates = [
        candidate for candidate in candidate_questions
        if candidate.strip().lower() != question.strip().lower()
    ]
    if not distinct_candidates:
        return None

    question_embedding = embedding_generator.embed_query(question)
    candidate_embeddings = embedding_generator.embed_texts(distinct_candidates)

    best_candidate = None
    best_score = similarity_threshold
    for candidate, embedding in zip(distinct_candidates, candidate_embeddings):
        score = _cosine_similarity(question_embedding, embedding)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate
