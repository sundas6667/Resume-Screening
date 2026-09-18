"""
ui/sidebar.py
==============
Persistent sidebar: identity, API key, sample CV generator, resume/JD
upload, and the Process button. Rendered once per script run from app.py
BEFORE the page navigation runs, so it's available on every page (Streamlit
re-runs the whole app.py script on every interaction, sidebar included,
regardless of which page is currently selected).
"""
from __future__ import annotations

from typing import List, Tuple

import streamlit as st

from config import APP_NAME, APP_TAGLINE, APP_VERSION, DEVELOPER, MAX_FILE_SIZE_MB, get_groq_api_key
from core.cv_generator import generate_sample_resume_pdf, list_available_profiles
from ui.pipeline import run_pipeline
from ui.session_state import reset_chatbot

_DEFAULT_JD_TEMPLATE = """Senior AI/ML Engineer

Requirements: 5+ years Python and ML experience, TensorFlow/PyTorch, NLP, Computer Vision, AWS/GCP, Docker, Kubernetes, MLOps.

Preferred: RAG systems, FastAPI, PostgreSQL, Team leadership."""


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.4rem 0 0.8rem 0;">
                <div style="display: inline-block; padding: 0.30rem 0.65rem; margin-bottom: 0.55rem;
                    border-radius: 999px; background: #CCFBF1; color: #0F766E;
                    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    AI Recruiter Suite
                </div>
                <h2 style="margin: 0; font-size: 1.7rem; font-weight: 800; color: #0F172A;">✨ {APP_NAME}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(APP_TAGLINE)
        st.divider()

        _render_api_key_input()
        st.divider()

        _render_sample_cv_generator()
        st.divider()

        uploaded_files = _render_resume_upload()
        jd_text = _render_job_description_input()
        st.divider()

        _render_process_button(uploaded_files, jd_text)

        st.divider()
        _render_footer()


def _render_api_key_input() -> None:
    st.markdown("**🔑 Groq API Key**")
    default_key = st.session_state.get("groq_api_key") or get_groq_api_key()
    key = st.text_input(
        "Groq API Key", value=default_key, type="password",
        label_visibility="collapsed", placeholder="gsk_...",
        help="Get a free key at console.groq.com. Kept only in this session, never written to disk.",
    )
    if key != st.session_state.get("groq_api_key"):
        st.session_state["groq_api_key"] = key
        reset_chatbot()
    if not key:
        st.caption("⚠️ Needed for the AI HR Assistant page.")


def _render_sample_cv_generator() -> None:
    st.markdown("**🧪 Try It: Sample CVs**")
    st.caption("No resumes handy? Generate realistic sample CVs to test the pipeline.")
    cols = st.columns(2)
    for i, profile in enumerate(list_available_profiles()):
        with cols[i % 2]:
            if st.button(profile["label"], key=f"gen_{profile['key']}", width="stretch"):
                pdf_bytes = generate_sample_resume_pdf(profile["key"])
                st.session_state["generated_sample_cvs"][f"{profile['label'].replace(' ', '_')}.pdf"] = pdf_bytes
                st.toast(f"Generated {profile['label']} sample CV — click Process below to include it.")
    generated = st.session_state.get("generated_sample_cvs", {})
    if generated:
        st.caption(f"📎 {len(generated)} sample CV(s) ready")
        if st.button("Clear sample CVs", key="clear_samples", width="stretch"):
            st.session_state["generated_sample_cvs"] = {}
            st.rerun()


def _render_resume_upload() -> List:
    st.markdown("**📄 Upload Resumes**")
    files = st.file_uploader(
        "Upload Resumes", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed", help=f"PDF only, up to {MAX_FILE_SIZE_MB}MB each.",
    )
    sample_count = len(st.session_state.get("generated_sample_cvs", {}))
    total = len(files or []) + sample_count
    if total:
        st.caption(f"📎 {len(files or [])} uploaded + {sample_count} sample = **{total}** resume(s) ready")
    return files or []


def _render_job_description_input() -> str:
    st.markdown("**💼 Job Description**")
    jd_source = st.radio(
        "JD source", ["Paste text", "Upload PDF"], label_visibility="collapsed", horizontal=True,
    )
    if jd_source == "Paste text":
        return st.text_area(
            "Job description", value=_DEFAULT_JD_TEMPLATE, height=180, label_visibility="collapsed",
        )

    jd_file = st.file_uploader("Upload JD PDF", type=["pdf"], label_visibility="collapsed", key="jd_pdf_upload")
    if jd_file is None:
        return ""
    from core.parser import ParsingError
    from ui.pipeline import get_parser
    try:
        doc = get_parser().parse(jd_file.getvalue(), jd_file.name)
        st.caption(f"✅ Loaded {len(doc.cleaned_text)} characters from {jd_file.name}")
        return doc.cleaned_text
    except ParsingError as exc:
        st.error(f"Could not read JD PDF: {exc}")
        return ""


def _render_process_button(uploaded_files: List, jd_text: str) -> None:
    sample_cvs = st.session_state.get("generated_sample_cvs", {})
    has_resumes = bool(uploaded_files or sample_cvs)
    has_jd = bool(jd_text and jd_text.strip())
    ready = has_resumes and has_jd

    if st.button("🚀 Process Resumes", type="primary", width="stretch", disabled=not ready):
        files_to_process: List[Tuple[bytes, str]] = [(f.getvalue(), f.name) for f in uploaded_files]
        files_to_process += [(pdf_bytes, filename) for filename, pdf_bytes in sample_cvs.items()]
        run_pipeline(files_to_process, jd_text)

    if not ready:
        missing = []
        if not has_resumes:
            missing.append("a resume")
        if not has_jd:
            missing.append("a job description")
        st.caption(f"Add {' and '.join(missing)} to continue.")


def _render_footer() -> None:
    st.markdown(f'<p class="app-footer">v{APP_VERSION} · {DEVELOPER}</p>', unsafe_allow_html=True)
