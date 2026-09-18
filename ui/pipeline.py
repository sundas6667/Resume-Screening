"""
ui/pipeline.py
================
Orchestrates the full parser -> extractor -> matcher -> ranking pipeline
with a Streamlit progress UI, storing results in session_state. This is UI
glue, not business logic: every step below just calls an already-built
core/ module  nothing here recomputes anything Modules 2-4 don't already do.
"""
from __future__ import annotations

from typing import List, Tuple

import streamlit as st

from core.extractor import ResumeExtractor
from core.matcher import compute_batch_statistics as compute_match_statistics, match_candidates
from core.parser import ResumeParser
from core.ranking import RankingEngine, compute_batch_statistics as compute_ranking_statistics, rank_candidates
from core.vector_store import VectorStore
from ui.session_state import reset_chatbot


@st.cache_resource(show_spinner=False)
def get_parser() -> ResumeParser:
    """Cached process-wide  cheap to construct, but sharing one instance
    avoids any doubt about state leaking between runs (there is none, but
    caching is still the idiomatic Streamlit pattern for a stateless service).
    """
    return ResumeParser()


@st.cache_resource(show_spinner=False)
def get_extractor() -> ResumeExtractor:
    """Cached process-wide  this is what makes the ~275ms SkillDatabase
    regex compilation a one-time cost rather than a per-run one.
    """
    return ResumeExtractor()


def run_pipeline(files: List[Tuple[bytes, str]], jd_text: str) -> None:
    """Run the full pipeline and store results in session_state. Never
    raises  any failure is caught and shown as a sidebar error, consistent
    with every core/ module's "never crash the app" contract.
    """
    parser = get_parser()
    extractor = get_extractor()
    weights = st.session_state["scoring_weights"]
    engine = RankingEngine(weights=weights)

    progress = st.sidebar.progress(0, text="Starting...")
    try:
        progress.progress(10, text="📤 Uploading & validating...")
        docs, failures = parser.parse_batch(files)

        candidates = []
        for i, doc in enumerate(docs):
            candidates.append(extractor.extract(doc))
            pct = 20 + int(35 * (i + 1) / max(len(docs), 1))
            progress.progress(pct, text=f"🔍 Parsing & extracting skills... ({i + 1}/{len(docs)})")

        progress.progress(58, text="💼 Analyzing job description...")
        job_description = extractor.extract_job_description(jd_text)

        progress.progress(68, text="🧮 Matching skills & computing semantic similarity...")
        vector_store = VectorStore()
        match_candidates(candidates, job_description, vector_store=vector_store)

        progress.progress(85, text="🏆 Ranking candidates...")
        ranked = rank_candidates(candidates, job_description, engine=engine)

        progress.progress(95, text="📊 Computing analytics...")
        match_stats = compute_match_statistics(ranked, job_description, vector_store=vector_store)
        ranking_stats = compute_ranking_statistics(ranked)

        st.session_state["candidates"] = ranked
        st.session_state["job_description"] = job_description
        st.session_state["vector_store"] = vector_store
        st.session_state["processing_errors"] = failures
        st.session_state["batch_stats_match"] = match_stats
        st.session_state["batch_stats_ranking"] = ranking_stats
        st.session_state["processing_complete"] = True
        st.session_state["selected_candidate_id"] = ranked[0].candidate_id if ranked else None
        reset_chatbot()

        progress.progress(100, text="✅ Analysis complete!")
        if failures and ranked:
            st.sidebar.warning(f"✅ {len(ranked)} processed · ⚠️ {len(failures)} failed — see Resume Screening.")
        elif failures and not ranked:
            st.sidebar.error(f"All {len(failures)} file(s) failed to process — see errors below.")
            for filename, message in failures:
                st.sidebar.caption(f"❌ {filename}: {message}")
        else:
            st.sidebar.success(f"✅ {len(ranked)} candidate(s) ranked successfully!")
    except Exception as exc:  # noqa: BLE001 - never crash the app on a pipeline failure
        st.sidebar.error(f"Processing failed unexpectedly: {exc}")
    finally:
        progress.empty()
