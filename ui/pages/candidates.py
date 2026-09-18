"""
ui/pages/candidates.py  👥 Candidates
Deep-dive on one candidate at a time: contact info, score breakdown radar
chart, skills, experience, education, and projects.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ui.components import render_empty_state
from ui.session_state import get_candidates, has_results


def render() -> None:
    st.title("👥 Candidate Profiles")

    if not has_results():
        render_empty_state("👥", "No candidates yet", "Process resumes from the sidebar to view candidate profiles.")
        return

    candidates = get_candidates()
    labels = [f"#{c.ranking.rank} {c.display_name}" for c in candidates]

    default_index = 0
    selected_id = st.session_state.get("selected_candidate_id")
    if selected_id:
        for i, c in enumerate(candidates):
            if c.candidate_id == selected_id:
                default_index = i
                break

    choice = st.selectbox("Select a candidate", labels, index=default_index)
    candidate = candidates[labels.index(choice)]
    st.session_state["selected_candidate_id"] = candidate.candidate_id

    _render_profile(candidate)


def _render_profile(candidate) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header(candidate.display_name)
        st.caption(f"{candidate.seniority_level or 'Experience level unknown'} · "
                   f"{candidate.total_years_experience:.1f} years experience")
        contact_bits = [x for x in [candidate.email, candidate.phone, candidate.location] if x]
        if contact_bits:
            st.write(" | ".join(contact_bits))
        if candidate.linkedin:
            url = candidate.linkedin if candidate.linkedin.startswith("http") else f"https://{candidate.linkedin}"
            st.markdown(f"🔗 [LinkedIn]({url})")
        if candidate.github:
            url = candidate.github if candidate.github.startswith("http") else f"https://{candidate.github}"
            st.markdown(f"💻 [GitHub]({url})")
    with col2:
        st.metric("Overall Score", f"{candidate.ranking.overall_score:.1f}/100")
        st.metric("Recommendation", candidate.ranking.recommendation or "—")
        confidence = candidate.diagnostics.confidence
        st.metric("Parsing Confidence", f"{confidence:.0f}%" if confidence is not None else "—")

    if candidate.diagnostics.warnings:
        st.warning("⚠️ " + " · ".join(candidate.diagnostics.warnings))

    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Score Breakdown", "🛠️ Skills", "💼 Experience", "🎓 Education", "🚀 Projects"]
    )

    with tab1:
        _render_radar_chart(candidate)
        if candidate.insights.recruiter_summary:
            st.markdown(f"**Recruiter Summary:** {candidate.insights.recruiter_summary}")

    with tab2:
        st.markdown("**Matched Skills:** " + (", ".join(candidate.matching.matched_skills) or "None"))
        st.markdown("**Missing Skills:** " + (", ".join(candidate.matching.missing_skills) or "None"))
        st.markdown("**All Detected Skills:** " + (", ".join(candidate.skills) or "None"))
        if candidate.certifications:
            st.markdown("**Certifications:** " + ", ".join(candidate.certifications))

    with tab3:
        if not candidate.experience:
            st.info("No experience entries extracted.")
        for exp in candidate.experience:
            title = f"{exp.designation or 'Role'} — {exp.company or 'Company'}"
            end = "Present" if exp.is_current else (exp.end_year or "?")
            st.markdown(f"**{title}** ({exp.start_year or '?'} - {end})")
            for responsibility in exp.responsibilities:
                st.markdown(f"- {responsibility}")

    with tab4:
        if not candidate.education:
            st.info("No education entries extracted.")
        for edu in candidate.education:
            degree_label = edu.degree_level or edu.degree or "Degree"
            field_suffix = f" in {edu.field_of_study}" if edu.field_of_study else ""
            st.markdown(f"**{degree_label}{field_suffix}**")
            year_suffix = f" · {edu.graduation_year}" if edu.graduation_year else ""
            st.caption(f"{edu.institution or 'Institution unknown'}{year_suffix}")

    with tab5:
        if not candidate.projects:
            st.info("No projects extracted.")
        for project in candidate.projects:
            st.markdown(f"**{project.name or 'Untitled Project'}**")
            if project.description:
                st.write(project.description)
            if project.tech_stack:
                st.caption("Tech: " + ", ".join(project.tech_stack))
            if project.github_link:
                st.caption(f"🔗 {project.github_link}")


def _render_radar_chart(candidate) -> None:
    breakdown = candidate.ranking.category_breakdown
    if not breakdown:
        st.info("No score breakdown available.")
        return
    categories = [category.title() for category in breakdown.keys()]
    values = [category_score.raw_score for category_score in breakdown.values()]

    figure = go.Figure()
    figure.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", name=candidate.display_name, line_color="#0F766E",
    ))
    figure.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E2E8F0"),
            angularaxis=dict(gridcolor="#E2E8F0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, height=350, margin=dict(t=30, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#0F172A"),
    )
    st.plotly_chart(figure, width="stretch")
