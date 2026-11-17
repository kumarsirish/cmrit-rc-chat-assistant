"""Shared page-title rendering: centered app icon + heading, reused by
every page (Chat, System Info, Admin, Recent Queries) for a consistent
look.

Lives in its own module rather than dashboard.py so admin_page.py and
rag._09_common.system_info.py can both import it too — dashboard.py
already imports render_admin_page from admin_page.py, so admin_page.py
importing back from dashboard.py would be circular.
"""

import base64
import html
from pathlib import Path

import streamlit as st

from rag._09_common.config import PROJECT_ROOT

APP_ICON_PATH = PROJECT_ROOT / "Pics" / "RC-Assistant.png"

# Displayed size (in pixels) for the page-title icon — the source
# image itself is much larger; this keeps it visually matched to the
# heading's size instead of rendering at some other arbitrary size.
APP_ICON_DISPLAY_WIDTH = 120


def render_page_title(title: str) -> None:
    """Shows the app icon and a heading centered as one group. Built as
    one inline-flex HTML block (not st.columns) so the pair is centered
    as a unit regardless of column-width rounding, and both items share
    the same vertical center line exactly."""
    icon_base64 = base64.b64encode(APP_ICON_PATH.read_bytes()).decode()
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:1rem; margin:1rem 0;">
            <img src="data:image/png;base64,{icon_base64}"
                 style="width:{APP_ICON_DISPLAY_WIDTH}px; height:{APP_ICON_DISPLAY_WIDTH}px; object-fit:contain;">
            <span style="font-size:2.5rem; font-weight:700; line-height:1;">{html.escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_centered_text(text: str) -> None:
    """A plain paragraph of text, center-justified — used under
    render_page_title for a page's intro/description line so it reads
    as one centered block with the heading above it."""
    st.markdown(f'<p style="text-align:center;">{html.escape(text)}</p>', unsafe_allow_html=True)
