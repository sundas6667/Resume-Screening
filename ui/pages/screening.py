"""
ui/pages/screening.py  📄 Resume Screening
The primary results view: color-coded ranking table + per-candidate
expandable strengths/weaknesses/missing-skills, with search/filter/sort.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SCORE_THRESHOLD_EXCELLENT, SCORE_THRESHOLD_GOOD
from ui.components import render_empty_state
from ui.session_state import get_candidates, has_results


def render() -> None:
    st.title("📄 Resume Screening")

    if not has_results():
        render_empty_state("📄", "No results yet", "Process resumes from the sidebar to see rankings here.")
        return

    candidates = get_candidates()
    errors = st.session_state.get("processing_errors", [])
    if errors:
        with st.expander(f"⚠️ {len(errors)} file(s) could not be processed", expanded=False):
            for filename, message in errors:
                st.markdown(f"- **{filename}**: {message}")

    col1, col2, col3 = st.columns([2, 1, 1])
    search = col1.text_input("🔍 Search by name or skill", "")
    sort_by = col2.selectbox("Sort by", ["Rank", "Overall Score", "Experience", "Name"])
    min_score = col3.slider("Minimum score", 0, 100, 0)

    filtered = _filter_candidates(candidates, search, min_score)
    filtered = _sort_candidates(filtered, sort_by)
    st.caption(f"Showing {len(filtered)} of {len(candidates)} candidates")

    _render_ranking_table(filtered)

    st.divider()
    st.subheader("Candidate Details")
    if not filtered:
        st.info("No candidates match the current filters.")
    for candidate in filtered:
        header = f"#{candidate.ranking.rank} {candidate.display_name} — {candidate.ranking.overall_score:.1f}/100 ({candidate.ranking.recommendation})"
        with st.expander(header):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Strengths**")
                for strength in candidate.insights.strengths:
                    st.markdown(f"- {strength}")
            with col2:
                st.markdown("**⚠️ Weaknesses**")
                for weakness in candidate.insights.weaknesses:
                    st.markdown(f"- {weakness}")
            st.markdown("**Missing skills:** " + (", ".join(candidate.matching.missing_skills) or "None"))
            st.caption(candidate.insights.recruiter_summary)


def _render_ranking_table(candidates) -> None:
    if not candidates:
        return
    rows = [{
        "Rank": c.ranking.rank, "Name": c.display_name, "Score": c.ranking.overall_score,
        "Recommendation": c.ranking.recommendation, "Skills Matched": len(c.matching.matched_skills),
        "Experience": f"{c.total_years_experience:.1f} yrs", "Confidence": c.ranking.confidence_level,
    } for c in candidates]
    df = pd.DataFrame(rows)

    def _color_score(value: float) -> str:
        if value >= SCORE_THRESHOLD_EXCELLENT:
            return "background-color: #D1FAE5; color: #166534"
        if value >= SCORE_THRESHOLD_GOOD:
            return "background-color: #FEF3C7; color: #92400E"
        return "background-color: #FEE2E2; color: #991B1B"

    styled = df.style.map(_color_score, subset=["Score"]).format({"Score": "{:.1f}"})
    st.dataframe(styled, width="stretch", hide_index=True)


def _filter_candidates(candidates, search: str, min_score: int):
    result = candidates
    if search:
        search_lower = search.lower()
        result = [
            c for c in result
            if search_lower in c.display_name.lower() or any(search_lower in s.lower() for s in c.skills)
        ]
    return [c for c in result if c.ranking.overall_score >= min_score]


def _sort_candidates(candidates, sort_by: str):
    if sort_by == "Overall Score":
        return sorted(candidates, key=lambda c: c.ranking.overall_score, reverse=True)
    if sort_by == "Experience":
        return sorted(candidates, key=lambda c: c.total_years_experience, reverse=True)
    if sort_by == "Name":
        return sorted(candidates, key=lambda c: c.display_name)
    return sorted(candidates, key=lambda c: c.ranking.rank or 999)
