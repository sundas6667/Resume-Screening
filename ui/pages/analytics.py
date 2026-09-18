"""
ui/pages/analytics.py — 📊 Analytics
Executive dashboard: score distribution, recommendation breakdown, most
common matched/missing skills, and per-category comparison. Reads only
from session_state  every chart here visualizes data Modules 2-4 already
computed.
"""
from __future__ import annotations

from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.components import render_empty_state
from ui.session_state import get_candidates, has_results

_CHART_FONT = dict(color="#0F172A")
_TRANSPARENT = "rgba(0,0,0,0)"


def render() -> None:
    st.title("📊 Analytics")

    if not has_results():
        render_empty_state("📊", "No data yet", "Process resumes from the sidebar to see analytics.")
        return

    candidates = get_candidates()
    ranking_stats = st.session_state.get("batch_stats_ranking")

    cols = st.columns(4)
    cols[0].metric("Highest Score", f"{ranking_stats.highest_score:.1f}" if ranking_stats else "—")
    cols[1].metric("Median Score", f"{ranking_stats.median_score:.1f}" if ranking_stats else "—")
    cols[2].metric("Lowest Score", f"{ranking_stats.lowest_score:.1f}" if ranking_stats else "—")
    cols[3].metric("Std. Deviation", f"{ranking_stats.std_deviation:.1f}" if ranking_stats else "—")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Comparison")
        _render_score_bar_chart(candidates)
    with col2:
        st.subheader("Recommendation Breakdown")
        _render_recommendation_pie(ranking_stats)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Common Matched Skills")
        _render_skill_frequency_chart(candidates, matched=True)
    with col2:
        st.subheader("Most Common Missing Skills")
        _render_skill_frequency_chart(candidates, matched=False)

    st.divider()
    st.subheader("Score Breakdown by Category")
    _render_category_comparison(candidates)


def _render_score_bar_chart(candidates) -> None:
    names = [c.display_name for c in candidates]
    scores = [c.ranking.overall_score for c in candidates]
    figure = px.bar(x=scores, y=names, orientation="h", labels={"x": "Overall Score", "y": ""})
    figure.update_traces(marker_color="#0F766E")
    figure.update_layout(paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT, font=_CHART_FONT, height=400)
    st.plotly_chart(figure, width="stretch")


def _render_recommendation_pie(ranking_stats) -> None:
    if not ranking_stats or not ranking_stats.recommendation_counts:
        st.info("No ranking data available.")
        return
    figure = go.Figure(data=[go.Pie(
        labels=list(ranking_stats.recommendation_counts.keys()),
        values=list(ranking_stats.recommendation_counts.values()), hole=0.45,
        marker=dict(colors=["#22C55E", "#14B8A6", "#F59E0B", "#EF4444"]),
    )])
    figure.update_layout(paper_bgcolor=_TRANSPARENT, font=_CHART_FONT, height=400)
    st.plotly_chart(figure, width="stretch")


def _render_skill_frequency_chart(candidates, matched: bool) -> None:
    skills = [
        skill for c in candidates
        for skill in (c.matching.matched_skills if matched else c.matching.missing_skills)
    ]
    top = Counter(skills).most_common(10)
    if not top:
        st.info("No data to display.")
        return
    color = "#22C55E" if matched else "#EF4444"
    figure = px.bar(x=[count for _, count in top], y=[skill for skill, _ in top], orientation="h")
    figure.update_traces(marker_color=color)
    figure.update_layout(
        paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT, font=_CHART_FONT, height=350,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(figure, width="stretch")


def _render_category_comparison(candidates) -> None:
    categories = ["skills", "experience", "education", "certification", "project"]
    figure = go.Figure()
    for candidate in candidates:
        breakdown = candidate.ranking.category_breakdown
        values = [breakdown[cat].raw_score if cat in breakdown else 0 for cat in categories]
        figure.add_trace(go.Bar(name=candidate.display_name, x=[cat.title() for cat in categories], y=values))
    figure.update_layout(
        barmode="group", paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT, font=_CHART_FONT, height=400,
    )
    st.plotly_chart(figure, width="stretch")
