from scripts.filter_low_faithfulness import filter_test_cases, select_low_faithfulness_questions


def test_select_low_faithfulness_questions_excludes_perfect_scores():
    rows = [
        {"question": "q1", "faithfulness": 1.0},
        {"question": "q2", "faithfulness": 0.5},
    ]
    assert select_low_faithfulness_questions(rows) == ["q2"]


def test_select_low_faithfulness_questions_includes_nan_scores():
    # RAGAS writes NaN when a judge call times out/errors rather than
    # raising, which is just as unverified as a low score.
    rows = [{"question": "q1", "faithfulness": float("nan")}]
    assert select_low_faithfulness_questions(rows) == ["q1"]


def test_select_low_faithfulness_questions_includes_missing_score():
    rows = [{"question": "q1"}]
    assert select_low_faithfulness_questions(rows) == ["q1"]


def test_select_low_faithfulness_questions_preserves_row_order():
    rows = [
        {"question": "q1", "faithfulness": 0.2},
        {"question": "q2", "faithfulness": 1.0},
        {"question": "q3", "faithfulness": 0.9},
    ]
    assert select_low_faithfulness_questions(rows) == ["q1", "q3"]


def test_filter_test_cases_keeps_only_matching_questions_in_original_order():
    test_cases = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
        {"question": "q3", "answer": "a3"},
    ]
    assert filter_test_cases(test_cases, ["q3", "q1"]) == [
        {"question": "q1", "answer": "a1"},
        {"question": "q3", "answer": "a3"},
    ]


def test_filter_test_cases_returns_empty_for_no_matches():
    test_cases = [{"question": "q1", "answer": "a1"}]
    assert filter_test_cases(test_cases, ["not-present"]) == []
