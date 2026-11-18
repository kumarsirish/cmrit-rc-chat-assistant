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

APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "RC-Assistant.png"

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
                 style="width:{APP_ICON_DISPLAY_WIDTH}px; height:{APP_ICON_DISPLAY_WIDTH}px;
                        object-fit:cover; border-radius:50%;
                        box-shadow:0 4px 14px rgba(37,99,235,0.25);
                        border:3px solid #ffffff;">
            <span style="font-size:2.5rem; font-weight:800; line-height:1;
                         color:#1E293B; letter-spacing:-0.02em;">{html.escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_centered_text(text: str) -> None:
    """A plain paragraph of text, center-justified — used under
    render_page_title for a page's intro/description line so it reads
    as one centered block with the heading above it."""
    st.markdown(
        f'<p style="text-align:center; color:#64748B; margin-top:-0.5rem;">{html.escape(text)}</p>',
        unsafe_allow_html=True,
    )


def render_status_pill(text: str) -> None:
    """A small rounded badge, centered, for one-line status/metadata
    (e.g. the estimated-answer-time figure) — a lighter-weight visual
    than a full st.info/st.caption block for a single transient fact."""
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin:0.25rem 0 0.75rem 0;">
            <span style="display:inline-block; padding:0.3rem 0.9rem; border-radius:999px;
                         background:#EEF2FF; color:#2563EB; font-size:0.8rem; font-weight:600;
                         border:1px solid #DBEAFE;">{html.escape(text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
