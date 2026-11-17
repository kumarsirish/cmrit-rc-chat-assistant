"""admin_page.py is otherwise Streamlit UI code with no test precedent
in this codebase (see dashboard.py/system_info.py, also untested) —
only the pure, non-Streamlit helper functions are covered here."""

from datetime import datetime

from rag._10_ui.admin_page import _format_timestamp, _resolve_category


def test_resolve_category_prefers_typed_over_selected():
    assert _resolve_category("Leave Policy", "Quality") == "Quality"


def test_resolve_category_falls_back_to_selected_when_nothing_typed():
    assert _resolve_category("Leave Policy", "") == "Leave Policy"
    assert _resolve_category("Leave Policy", "   ") == "Leave Policy"


def test_format_timestamp_uses_dd_mm_yyyy_hh_mm_ss():
    unix_timestamp = datetime(2026, 3, 5, 14, 30, 45).timestamp()
    assert _format_timestamp(unix_timestamp) == "05-03-2026, 14:30:45"
