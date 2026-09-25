from rag.routing.query_expander import build_glossary_index, expand_query

GLOSSARY = {
    "RCPIE": "research consultancy projects innovation and entrepreneurship",
    "IQAC": "internal quality assurance cell",
}


def test_expand_query_appends_matched_acronym():
    index = build_glossary_index(GLOSSARY)
    expanded = expand_query("what is the research consultancy load?", GLOSSARY, index)
    assert "RCPIE" in expanded
    assert "what is the research consultancy load?" in expanded


def test_expand_query_returns_unchanged_when_no_match():
    index = build_glossary_index(GLOSSARY)
    question = "how many casual leaves am I allowed per year?"
    assert expand_query(question, GLOSSARY, index) == question


def test_expand_query_requires_at_least_two_matched_words():
    index = build_glossary_index(GLOSSARY)
    # "quality" alone is a single-word overlap with IQAC's phrase and
    # must not be enough to pull in an unrelated acronym expansion.
    question = "what is CMRIT's quality policy"
    assert expand_query(question, GLOSSARY, index) == question


def test_expand_query_matches_multiple_acronyms():
    index = build_glossary_index(GLOSSARY)
    expanded = expand_query("who handles internal quality assurance and research consultancy?", GLOSSARY, index)
    assert "RCPIE" in expanded
    assert "IQAC" in expanded


def test_expand_query_returns_unchanged_when_question_already_has_acronym():
    index = build_glossary_index(GLOSSARY)
    question = "what is RCPIE?"
    expanded = expand_query(question, GLOSSARY, index)
    assert expanded == question or expanded.startswith(question)


def test_expand_query_handles_empty_glossary():
    question = "how much is the consultancy load?"
    assert expand_query(question, {}, {}) == question


def test_build_glossary_index_skips_stopwords_and_short_words():
    index = build_glossary_index({"XYZ": "the of a is"})
    assert index == {}


def test_build_glossary_index_drops_words_shared_by_multiple_acronyms():
    glossary = {
        "FDP": "faculty development program",
        "DMS": "document management program",
    }
    index = build_glossary_index(glossary)
    # "program" occurs in both phrases, so it's too generic to point at
    # either acronym and must be dropped rather than indexed.
    assert "program" not in index
    assert index["faculty"] == {"FDP"}
    assert index["development"] == {"FDP"}
    assert index["document"] == {"DMS"}
    assert index["management"] == {"DMS"}
