"""Step 4b: Institutional glossary extraction.

Builds an acronym -> full-phrase glossary automatically from the
ingested documents themselves, instead of a hand-maintained keyword
list per topic (see rag._03_routing.docs_metadata.DocTopic.keywords for
that older, per-topic approach). Institutional documents almost
always spell an acronym's meaning out on first use, in one of two
conventional forms: "<full phrase> (<ACRONYM>)" or "<ACRONYM> (<full
phrase>)" — this module regex-matches both forms and keeps only the
ones where the acronym's letters are actually the initials of the
captured phrase, which discards the vast majority of parenthetical
text that isn't really a definition.

Used by rag._03_routing.query_expander to fold this vocabulary into the
retrieval query before topic-matching/embedding — see that module for
why this has to happen before retrieval, not just in the answer LLM's
system prompt.
"""

import json
import re
from collections import Counter

# Matches "<phrase> (<ACRONYM>)", e.g. "Internal Quality Assurance Cell (IQAC)".
_PHRASE_THEN_ACRONYM = re.compile(r"([A-Za-z][A-Za-z,&\-\s]{3,150}?)\s*\(\s*([A-Z]{2,8})\s*\)")
# Matches "<ACRONYM> (<phrase>)", e.g. "IQAC (Internal Quality Assurance Cell)".
_ACRONYM_THEN_PHRASE = re.compile(r"\b([A-Z]{2,8})\s*\(\s*([A-Za-z][A-Za-z,&\-\s]{3,150}?)\s*\)")

# Skipped when computing a phrase's initials, so a captured span with
# an incidental leading/embedded filler word (regex match boundaries
# aren't sentence-aware, so this is common) can still match the
# acronym it actually defines, e.g. "and Special Duty Leave" -> "SDL".
_FILLER_WORDS = {"and", "of", "the", "for", "&", "to", "in", "an", "a", "at", "on", "with"}

# Acronyms shorter than this collide too often with unrelated phrases
# to be a useful, unambiguous glossary entry — observed in this corpus:
# "CO" matched both "Compensatory off" and "Course Outcomes".
MIN_ACRONYM_LENGTH = 3


def _best_suffix_match(acronym: str, phrase: str) -> str:
    """Trims phrase from the left, word by word, and returns the
    shortest trailing word sequence whose initials equal acronym, or
    None if no such suffix exists.

    The regexes above can't know where a definition's phrase actually
    starts — that's not decidable from punctuation alone — so they
    often capture extra leading context, e.g. "the faculty at CMRIT
    pursue research, consultancy, projects, innovation and
    entrepreneurship" for "RCPIE". Requiring the full captured span to
    match would reject this real definition; trimming from the left
    until the initials line up recovers it.
    """
    words = [w for w in re.split(r"[\s,&/-]+", phrase.strip()) if w]
    for start in range(len(words)):
        candidate_words = words[start:]
        significant = [w for w in candidate_words if w.lower() not in _FILLER_WORDS]
        if not significant:
            continue
        initials = "".join(w[0] for w in significant).upper()
        if initials == acronym:
            # Drop any filler word(s) left at the very front of the
            # match (e.g. "The Internal Quality Assurance Cell" for
            # "IQAC" — "The" isn't part of the initials but survived
            # because trimming only advances start on a failed match,
            # not on a leading filler word within a successful one).
            while candidate_words and candidate_words[0].lower() in _FILLER_WORDS:
                candidate_words = candidate_words[1:]
            return " ".join(candidate_words)
    return None


def _candidates_from_text(text: str) -> list:
    text = re.sub(r"[ \t]+", " ", text)
    found = []
    for phrase, acronym in _PHRASE_THEN_ACRONYM.findall(text):
        acronym = acronym.upper()
        match = _best_suffix_match(acronym, phrase)
        if match:
            found.append((acronym, match))
    for acronym, phrase in _ACRONYM_THEN_PHRASE.findall(text):
        acronym = acronym.upper()
        match = _best_suffix_match(acronym, phrase)
        if match:
            found.append((acronym, match))
    return found


def extract_glossary(pages: list, min_acronym_length: int = MIN_ACRONYM_LENGTH) -> dict:
    """Scans every page's text for acronym definitions and returns an
    {ACRONYM: phrase} glossary, one entry per acronym.

    Where the same acronym is defined more than once (repeated in one
    document, or phrased slightly differently across documents), the
    most frequently seen phrasing wins; a mixed-case rendering is
    preferred over an ALL-CAPS one when both occur equally often,
    since the all-caps form is usually just a heading rendering of the
    same phrase rather than a distinct definition.
    """
    raw: dict = {}
    for page in pages:
        for acronym, phrase in _candidates_from_text(page.text):
            if len(acronym) < min_acronym_length:
                continue
            raw.setdefault(acronym, []).append(phrase)

    glossary = {}
    for acronym, phrases in raw.items():
        best_casefold, _ = Counter(p.casefold() for p in phrases).most_common(1)[0]
        display_candidates = [p for p in phrases if p.casefold() == best_casefold]
        non_upper = [p for p in display_candidates if not p.isupper()]
        glossary[acronym] = (non_upper or display_candidates)[0]
    return glossary


def save_glossary(glossary: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as glossary_file:
        json.dump(dict(sorted(glossary.items())), glossary_file, indent=2, ensure_ascii=False)
        glossary_file.write("\n")


def load_glossary(path: str) -> dict:
    """Returns {} if path doesn't exist yet (e.g. ingestion hasn't run
    since this module was added) rather than raising — the glossary is
    an optional retrieval enhancement, not a hard dependency."""
    try:
        with open(path, "r", encoding="utf-8") as glossary_file:
            return json.load(glossary_file)
    except FileNotFoundError:
        return {}
