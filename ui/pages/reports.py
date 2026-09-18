"""
ui/pages/reports.py  📑 Reports
Export center: CSV/Excel/PDF/JSON, generated purely from already-computed
data — this page recomputes nothing, only calls core/report_generator.py.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from config import APP_NAME
from core.report_generator import (
    export_to_csv,
    export_to_excel,
    export_to_json,
    export_to_pdf,
    generate_export_manifest,
)
from ui.components import render_empty_state
from ui.session_state import get_candidates, get_job_description, has_results


def render() -> None:
    st.title("📑 Export Center")

    if not has_results():
        render_empty_state("📑", "Nothing to export yet", "Process resumes from the sidebar first.")
        return

    candidates = get_candidates()
    job_description = get_job_description()
    st.caption(f"{len(candidates)} candidate(s) ready to export.")

    with st.expander("🏷️ Report branding (optional)"):
        company_name = st.text_input("Company / product name", value=APP_NAME)
        recruiter_name = st.text_input("Prepared by (optional)", value="")
        report_title = st.text_input("Report title", value="Candidate Ranking Report")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**📄 PDF Report**")
        st.caption("Executive summary + ranking table + per-candidate breakdown.")
        pdf_bytes = export_to_pdf(
            candidates, job_description=job_description, company_name=company_name,
            recruiter_name=recruiter_name or None, report_title=report_title,
        )
        st.download_button(
            "Download PDF", pdf_bytes, file_name=f"candidate_report_{timestamp}.pdf",
            mime="application/pdf", width="stretch",
        )

    with col2:
        st.markdown("**📊 Excel Workbook**")
        st.caption("Color-coded ranking spreadsheet.")
        excel_bytes = export_to_excel(candidates)
        st.download_button(
            "Download Excel", excel_bytes, file_name=f"candidate_rankings_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    with col3:
        st.markdown("**📋 CSV File**")
        st.caption("Plain spreadsheet-compatible export.")
        csv_bytes = export_to_csv(candidates)
        st.download_button(
            "Download CSV", csv_bytes, file_name=f"candidate_rankings_{timestamp}.csv",
            mime="text/csv", width="stretch",
        )

    with col4:
        st.markdown("**🗂️ JSON Export**")
        st.caption("Full structured data, for developers.")
        json_bytes = export_to_json(candidates, job_description=job_description)
        st.download_button(
            "Download JSON", json_bytes, file_name=f"candidate_data_{timestamp}.json",
            mime="application/json", width="stretch",
        )

    st.divider()
    manifest_bytes = generate_export_manifest(candidates, ["pdf", "xlsx", "csv", "json"])
    st.download_button(
        "📜 Download export manifest", manifest_bytes, file_name=f"export_manifest_{timestamp}.json",
        mime="application/json",
    )
