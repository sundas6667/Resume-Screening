"""
ui/session_state.py
=====================
Typed session_state initialization and accessors  the single place that
knows what keys exist in st.session_state, so pages don't each invent their
own defaults or risk a KeyError on first load.
"""
from __future__ import annotations

from typing import List, Optional

import streamlit as st

from config import ScoringWeights
from core.chat.chatbot import HRChatbot
from models.candidate import Candidate
from models.job_description import JobDescription

_DEFAULTS = {
    "candidates": [],
    "job_description": None,
    "vector_store": None,
    "groq_api_key": "",
    "scoring_weights": None,  # set lazily via ScoringWeights.load() below
    "processing_complete": False,
    "processing_errors": [],
    "chatbot": None,
    "selected_candidate_id": None,
    "batch_stats_match": None,
    "batch_stats_ranking": None,
    "generated_sample_cvs": {},
}


def init_session_state() -> None:
    """Call once at the top of app.py, before rendering anything."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state["scoring_weights"] is None:
        st.session_state["scoring_weights"] = ScoringWeights.load()


def get_candidates() -> List[Candidate]:
    return st.session_state.get("candidates", [])


def get_job_description() -> Optional[JobDescription]:
    return st.session_state.get("job_description")


def has_results() -> bool:
    return bool(st.session_state.get("processing_complete")) and bool(get_candidates())


def get_or_create_chatbot() -> Optional[HRChatbot]:
    """Lazily construct the chatbot once candidates + an API key are
    available, reusing the existing VectorStore (built during matching)
    rather than constructing a new one.
    """
    if not st.session_state.get("groq_api_key"):
        return None
    if not get_candidates():
        return None
    existing = st.session_state.get("chatbot")
    if existing is not None:
        return existing
    try:
        bot = HRChatbot(
            api_key=st.session_state["groq_api_key"],
            candidates=get_candidates(),
            job_description=get_job_description(),
            vector_store=st.session_state.get("vector_store"),
        )
    except Exception:  # noqa: BLE001 - e.g. ChatError for a missing/invalid key
        return None
    st.session_state["chatbot"] = bot
    return bot


def reset_chatbot() -> None:
    """Called after reprocessing (new candidates/JD) or an API key change,
    so the chatbot picks up fresh data instead of talking about stale
    candidates or using a stale client.
    """
    st.session_state["chatbot"] = None


def get_selected_candidate() -> Optional[Candidate]:
    candidate_id = st.session_state.get("selected_candidate_id")
    if not candidate_id:
        return None
    return next((c for c in get_candidates() if c.candidate_id == candidate_id), None)
