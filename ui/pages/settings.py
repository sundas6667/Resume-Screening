"""
ui/pages/settings.py  ⚙️ Settings
Weight tuning re-ranks in place using the already-matched candidates — it
never re-parses or re-matches (see core.matcher.match_candidates, which is
NOT called here), so applying new weights is fast and consistent with the
rest of the app. Also surfaces app/model version info.
"""
from __future__ import annotations

import streamlit as st

from config import APP_NAME, APP_VERSION, DEVELOPER, GROQ_MODEL, MODEL_VERSION, SCHEMA_VERSION, ScoringWeights
from core.ranking import RankingEngine, compute_batch_statistics as compute_ranking_statistics, rank_candidates
from ui.session_state import get_candidates, get_job_description, has_results


def render() -> None:
    st.title("⚙️ Settings")
    tab1, tab2 = st.tabs(["⚖️ Scoring Weights", "ℹ️ About"])
    with tab1:
        _render_weights_tab()
    with tab2:
        _render_about_tab()


def _render_weights_tab() -> None:
    st.subheader("Category Weights")
    st.caption("Adjust how much each category contributes to the overall score. Must sum to 100%.")

    weights: ScoringWeights = st.session_state["scoring_weights"]
    col1, col2 = st.columns(2)
    with col1:
        skills = st.slider("Skills", 0, 100, int(round(weights.skills * 100)))
        experience = st.slider("Experience", 0, 100, int(round(weights.experience * 100)))
        education = st.slider("Education", 0, 100, int(round(weights.education * 100)))
    with col2:
        certification = st.slider("Certification", 0, 100, int(round(weights.certification * 100)))
        project = st.slider("Project", 0, 100, int(round(weights.project * 100)))

    total = skills + experience + education + certification + project
    if total != 100:
        st.error(f"Weights must sum to 100% (currently {total}%).")
    else:
        st.success("✅ Weights sum to 100%.")

    if st.button("Apply & Re-rank", disabled=(total != 100), type="primary"):
        new_weights = ScoringWeights(
            skills=skills / 100, experience=experience / 100, education=education / 100,
            certification=certification / 100, project=project / 100,
        )
        st.session_state["scoring_weights"] = new_weights
        try:
            new_weights.save()
        except ValueError:
            pass  # validated above; defensive only

        if has_results():
            engine = RankingEngine(weights=new_weights)
            ranked = rank_candidates(get_candidates(), get_job_description(), engine=engine)
            st.session_state["candidates"] = ranked
            st.session_state["batch_stats_ranking"] = compute_ranking_statistics(ranked)
            st.success("Re-ranked with the new weights — see Resume Screening or Dashboard.")
        else:
            st.info("Weights saved — they'll apply the next time you process resumes.")


def _render_about_tab() -> None:
    st.subheader("About")
    st.markdown(f"**{APP_NAME}** · v{APP_VERSION}")
    st.caption(f"Developed by {DEVELOPER}")
    st.divider()
    st.markdown(f"""
| | |
|---|---|
| AI Model (Chatbot) | `{GROQ_MODEL}` |
| Model/Extraction Version | `{MODEL_VERSION}` |
| Data Schema Version | `{SCHEMA_VERSION}` |
""")
    st.caption(
        "Built with Streamlit, scikit-learn (TF-IDF + cosine similarity), PyPDF2, "
        "reportlab, openpyxl, and Groq — no PyTorch, FAISS, or GPU dependencies."
    )
