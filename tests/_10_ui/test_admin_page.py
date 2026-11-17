"""admin_page.py is otherwise Streamlit UI code with no test precedent
in this codebase (see dashboard.py/system_info.py, also untested) —
only the pure, non-Streamlit helper functions are covered here."""

from rag._10_ui.admin_page import _existing_categories, _resolve_category


def test_resolve_category_prefers_typed_over_selected():
    assert _resolve_category("Leave Policy", "Quality") == "Quality"


def test_resolve_category_falls_back_to_selected_when_nothing_typed():
    assert _resolve_category("Leave Policy", "") == "Leave Policy"
    assert _resolve_category("Leave Policy", "   ") == "Leave Policy"


def test_existing_categories_lists_subfolders_sorted(tmp_path):
    (tmp_path / "Quality").mkdir()
    (tmp_path / "Leave Policy").mkdir()
    (tmp_path / "not_a_dir.pdf").write_bytes(b"x")

    assert _existing_categories(str(tmp_path)) == ["Leave Policy", "Quality"]


def test_existing_categories_returns_empty_list_when_directory_missing(tmp_path):
    assert _existing_categories(str(tmp_path / "missing")) == []
