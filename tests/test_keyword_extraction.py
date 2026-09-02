from rag.keyword_extraction import extract_keywords


def test_extracts_significant_words_and_drops_stopwords():
    assert extract_keywords("what is darwinbox") == ["darwinbox"]


def test_lowercases_and_dedupes_in_first_seen_order():
    assert extract_keywords("Leave Leave policy LEAVE") == ["leave", "policy"]


def test_drops_short_words():
    assert extract_keywords("is it ok to go home") == ["home"]


def test_all_stopwords_returns_empty_list():
    assert extract_keywords("what is it") == []


def test_keeps_hyphenated_and_alphanumeric_words():
    assert extract_keywords("What is a work-from-home policy?") == [
        "work-from-home",
        "policy",
    ]
