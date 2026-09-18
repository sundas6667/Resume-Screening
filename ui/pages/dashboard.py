"""
ui/pages/dashboard.py  🏠 Dashboard
Landing page: empty state before processing, top-line stats + quick
ranking preview after. Reads only from session_state; computes nothing.
"""
from __future__ import annotations

import streamlit as st

from ui.components import render_empty_state, score_badge_html
from ui.session_state import get_candidates, get_job_description, has_results


def render() -> None:
    st.title("🏠 Dashboard")

    if not has_results():
        render_empty_state(
            "📄", "No candidates processed yet",
            "Upload resumes and a job description in the sidebar, then click <b>Process Resumes</b> "
            "to get started. No resumes handy? Try the sample CV generator in the sidebar.",
        )
        return

    candidates = get_candidates()
    job_description = get_job_description()
    match_stats = st.session_state.get("batch_stats_match")
    ranking_stats = st.session_state.get("batch_stats_ranking")

    if job_description and job_description.title:
        st.caption(f"Screening for: **{job_description.title}**")

    cols = st.columns(6)
    cols[0].metric("Total Candidates", ranking_stats.candidate_count if ranking_stats else len(candidates))
    cols[1].metric("Average Score", f"{ranking_stats.average_score:.1f}" if ranking_stats else "—")
    top_candidate = candidates[0] if candidates else None
    cols[2].metric("Top Candidate", top_candidate.display_name if top_candidate else "—")
    avg_years = sum(c.total_years_experience for c in candidates) / len(candidates) if candidates else 0.0
    cols[3].metric("Avg. Experience", f"{avg_years:.1f} yrs")
    cols[4].metric("Avg. Skill Coverage", f"{match_stats.average_skill_coverage:.0f}%" if match_stats else "—")
    total_time_ms = sum(c.metadata.processing_time_ms for c in candidates)
    cols[5].metric("Processing Time", f"{total_time_ms:.0f}ms")

    st.divider()
    st.subheader("Top Candidates")
    header = st.columns([3, 1.2, 2, 3])
    header[0].markdown("**Candidate**")
    header[1].markdown("**Score**")
    header[2].markdown("**Recommendation**")
    header[3].markdown("**Top Matched Skills**")

    for candidate in candidates[:5]:
        row = st.columns([3, 1.2, 2, 3])
        row[0].markdown(f"**#{candidate.ranking.rank} {candidate.display_name}**")
        row[1].markdown(score_badge_html(candidate.ranking.overall_score), unsafe_allow_html=True)
        row[2].markdown(candidate.ranking.recommendation or "—")
        row[3].markdown(", ".join(candidate.matching.matched_skills[:3]) or "—")

    st.caption(
        "See **Resume Screening** for the full ranking table and filters, or **Candidates** "
        "for individual profiles."
    )
