"""Pulls the meaningful words out of a short query.

Used by the retriever as a fallback when semantic search alone finds
nothing above the relevance threshold: a handful of significant words
(proper nouns, brand names, policy terms) can still be searched for
as a literal substring, even when the embedding model has too little
signal to place a short, name-heavy query semantically.
"""

import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "whom", "which", "how", "why", "when", "where",
    "do", "does", "did", "doing", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "for", "of", "to", "in", "on", "at", "by", "with", "about", "as",
    "and", "or", "but", "if", "so", "this", "that", "these", "those",
    "tell", "explain", "please",
}

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]*")


def extract_keywords(text: str) -> list:
    """Returns the significant (non-stopword) words in text, lowercased,
    in first-appearance order, with duplicates removed.
    """
    keywords = []
    seen = set()
    for match in _WORD_PATTERN.finditer(text.lower()):
        word = match.group()
        if word in _STOPWORDS or len(word) < 3:
            continue
        if word not in seen:
            seen.add(word)
            keywords.append(word)
    return keywords
