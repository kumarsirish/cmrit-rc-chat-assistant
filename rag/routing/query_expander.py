"""Expands a user's question with institutional-glossary terms before
it's used for topic-matching/dense retrieval (see rag_pipeline.py).

The pipeline is strictly grounded in retrieved chunks — the answer LLM
never sees any vocabulary the retrieval step didn't already surface as
context. So bridging a jargon gap (e.g. a question saying "consultancy
load" when the actual documents/topic are keyed to "RCPIE") has to
happen before/during retrieval, by rewriting the query itself, not by
teaching the answer LLM a glossary it has no way to act on.

This expander is a cheap, deterministic word-overlap lookup rather
than an LLM call: build_glossary_index() maps each significant word in
a glossary phrase to the acronym(s) it defines, and expand_query()
appends any acronym whose defining phrase shares a word with the
question. No extra network/LLM round trip, so it costs nothing on the
hot path — see rag.ingestion.glossary_extractor for where the
glossary itself comes from.
"""

import re
from collections import defaultdict

_WORD_PATTERN = re.compile(r"[A-Za-z']+")

# Skipped when indexing/matching so a common word inside a glossary
# phrase (e.g. "and", "of") doesn't make every question containing it
# spuriously match every acronym whose phrase happens to include it.
_STOPWORDS = {
    "and", "of", "the", "for", "to", "in", "an", "a", "at", "on", "with",
    "is", "are", "was", "were", "be", "been", "or", "as", "by", "from",
}
# Below this length, a word is too common/short to be a reliable
# glossary-match signal (e.g. "an", "co") — filtered on both the
# glossary side (index) and the query side (lookup).
_MIN_WORD_LENGTH = 4

# A question must share at least this many significant words with a
# glossary phrase before that acronym is expanded in. A single shared
# word is too weak a signal on its own — e.g. "quality" alone would
# otherwise match IQAC ("Internal Quality Assurance Cell") for any
# question mentioning "quality policy", pulling retrieval toward IQAC
# content instead of the unrelated policy actually being asked about.
_MIN_MATCHED_WORDS = 2


def _significant_words(text: str) -> set:
    words = _WORD_PATTERN.findall(text.lower())
    return {word for word in words if word not in _STOPWORDS and len(word) >= _MIN_WORD_LENGTH}


def build_glossary_index(glossary: dict) -> dict:
    """{significant word -> set of acronyms whose glossary phrase
    contains that word}, built once (not per-query) so expand_query
    does O(words in question) dict lookups instead of scanning the
    whole glossary for every question.

    Words that occur in more than one glossary phrase (e.g.
    "management", "development") are dropped entirely rather than
    indexed: shared across many acronyms, they're generic enough that
    matching on them doesn't distinguish which acronym (if any) the
    question is actually about.
    """
    word_acronyms = defaultdict(set)
    for acronym, phrase in glossary.items():
        for word in _significant_words(phrase):
            word_acronyms[word].add(acronym)

    index = defaultdict(set)
    for word, acronyms in word_acronyms.items():
        if len(acronyms) == 1:
            index[word].update(acronyms)
    return dict(index)


def expand_query(question: str, glossary: dict, glossary_index: dict) -> str:
    """Returns question with any matched acronyms' full phrases
    appended, for use as the retrieval query only — the original
    question (not this expanded form) is still what's shown to the
    user and sent to the answer LLM, so citations/phrasing stay clean
    and unaffected by this.

    An acronym is only expanded in once its phrase shares at least
    _MIN_MATCHED_WORDS significant words with the question — see that
    constant's docstring for why a single shared word isn't enough.

    A question that already contains the acronym itself (someone who
    already knows to say "RCPIE") matches nothing new here and is
    returned unchanged — this only helps the jargon-gap case.
    """
    if not glossary or not glossary_index:
        return question

    match_counts = defaultdict(int)
    for word in _significant_words(question):
        for acronym in glossary_index.get(word, ()):
            match_counts[acronym] += 1

    matched_acronyms = {acronym for acronym, count in match_counts.items() if count >= _MIN_MATCHED_WORDS}

    if not matched_acronyms:
        return question

    additions = " ".join(f"{acronym} ({glossary[acronym]})" for acronym in sorted(matched_acronyms))
    return f"{question}\n\n{additions}"
