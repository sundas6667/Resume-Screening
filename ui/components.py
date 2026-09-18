"""
ui/components.py
==================
Small reusable UI pieces used across multiple pages, kept here so styling
stays consistent instead of each page reinventing its own badge/empty-state
HTML.
"""
from __future__ import annotations

import streamlit as st

from config import SCORE_THRESHOLD_EXCELLENT, SCORE_THRESHOLD_GOOD


def score_badge_class(score: float) -> str:
    if score >= SCORE_THRESHOLD_EXCELLENT:
        return "score-badge-excellent"
    if score >= SCORE_THRESHOLD_GOOD:
        return "score-badge-good"
    return "score-badge-poor"


def score_badge_html(score: float, label: str = "") -> str:
    """HTML for a colored score pill — wrap the caller's markdown in
    unsafe_allow_html=True.
    """
    text = label or f"{score:.0f}%"
    return f'<span class="score-badge {score_badge_class(score)}">{text}</span>'


def render_empty_state(icon: str, title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="app-empty-state">
            <div class="app-empty-state-icon">{icon}</div>
            <h3>{title}</h3>
            <p class="app-empty-state-message">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
