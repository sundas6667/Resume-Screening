"""
core/report_generator.py
==========================
Turns already computed Candidate/MatchingResult/RankingResult data into
downloadable reports: CSV, Excel, PDF, and JSON. This module recomputes
NOTHING — every number here was already produced by core/ranking.py; this
is purely presentation, so exports stay consistent with whatever the
dashboard and chatbot show for the same candidates.
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import List, Optional
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import APP_NAME, SCORE_THRESHOLD_EXCELLENT, SCORE_THRESHOLD_GOOD, THEME
from models.candidate import Candidate
from models.job_description import JobDescription
from utils.helpers import get_logger

logger = get_logger(__name__)

_GREEN_HEX = "C6EFCE"
_YELLOW_HEX = "FFEB9C"
_RED_HEX = "FFC7CE"


def _score_band_hex(score: float) -> str:
    if score >= SCORE_THRESHOLD_EXCELLENT:
        return _GREEN_HEX
    if score >= SCORE_THRESHOLD_GOOD:
        return _YELLOW_HEX
    return _RED_HEX


def _esc(value: object) -> str:
    """Escape text for reportlab's Paragraph (which parses a small XML-like
    markup)  resume/JD content can contain '&', '<', etc. and must not be
    allowed to break report generation.
    """
    return xml_escape(str(value))


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def export_to_csv(candidates: List[Candidate]) -> bytes:
    """One row per candidate, via Candidate.to_export_row() , the single
    source of truth for "what a spreadsheet row looks like" (also used by
    the Excel export below), so the two formats never drift apart.
    """
    if not candidates:
        return "No candidates to export.\n".encode("utf-8")
    df = pd.DataFrame([c.to_export_row() for c in candidates])
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def export_to_json(candidates: List[Candidate], job_description: Optional[JobDescription] = None) -> bytes:
    """Full structured export  one full Candidate.to_json() per candidate,
    plus JD context. The richest export format; CSV/Excel are summaries.
    """
    payload = {
        "generated_at": datetime.now().isoformat(),
        "app_name": APP_NAME,
        "job_description": (
            {
                "title": job_description.title,
                "required_skills": job_description.required_skills,
                "preferred_skills": job_description.preferred_skills,
                "min_years_experience": job_description.min_years_experience,
                "required_education_level": job_description.required_education_level,
            }
            if job_description else None
        ),
        "candidate_count": len(candidates),
        "candidates": [c.to_json() for c in candidates],
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def export_to_excel(candidates: List[Candidate]) -> bytes:
    """Color-coded ranking sheet (green/yellow/red on Overall Score, matching
    the same thresholds used everywhere else in the app) via openpyxl.
    """
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Rankings"

    if not candidates:
        sheet.append(["No candidates to export"])
        buffer = io.BytesIO()
        workbook.save(buffer)
        return buffer.getvalue()

    rows = [c.to_export_row() for c in candidates]
    headers = list(rows[0].keys())
    sheet.append(headers)

    header_fill = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    score_col_index = headers.index("Overall Score") + 1
    for row_dict in rows:
        sheet.append([row_dict[h] for h in headers])
        score_cell = sheet.cell(row=sheet.max_row, column=score_col_index)
        score_cell.fill = PatternFill(
            start_color=_score_band_hex(row_dict["Overall Score"]),
            end_color=_score_band_hex(row_dict["Overall Score"]),
            fill_type="solid",
        )

    for i in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(i)].width = 16
    sheet.freeze_panes = "A2"

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def export_to_pdf(
    candidates: List[Candidate],
    job_description: Optional[JobDescription] = None,
    company_name: str = APP_NAME,
    recruiter_name: Optional[str] = None,
    report_title: str = "Candidate Ranking Report",
) -> bytes:
    """Multi-page PDF report: title + executive summary, a color-coded
    ranking table, then one detail page per candidate (score breakdown,
    strengths, weaknesses, summary), with a reproducibility footer citing
    the scoring version and weights used.
    """
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=colors.HexColor(THEME["primary"]))
    section_style = ParagraphStyle(
        "ReportSection", parent=styles["Heading2"], textColor=colors.HexColor(THEME["primary"]),
        spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle("ReportBody", parent=styles["Normal"], fontSize=10, leading=13)

    elements = [
        Paragraph(_esc(company_name), title_style),
        Paragraph(_esc(report_title), styles["Heading2"]),
    ]
    if job_description and job_description.title:
        elements.append(Paragraph(f"Position: {_esc(job_description.title)}", body_style))
    if recruiter_name:
        elements.append(Paragraph(f"Prepared by: {_esc(recruiter_name)}", body_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", body_style))
    elements.append(Spacer(1, 0.25 * inch))

    if not candidates:
        elements.append(Paragraph("No candidates were available to include in this report.", body_style))
        document.build(elements)
        return buffer.getvalue()

    elements.append(_build_executive_summary(candidates, section_style, body_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(_build_ranking_table(candidates))
    elements.append(_build_footer_paragraph(candidates, body_style))
    elements.append(PageBreak())

    for candidate in candidates:
        elements.extend(_build_candidate_detail(candidate, section_style, body_style))
        elements.append(PageBreak())

    document.build(elements)
    return buffer.getvalue()


def _build_executive_summary(candidates: List[Candidate], section_style, body_style) -> Table:
    """One-page-friendly snapshot before the detail pages: batch stats plus
    a recommendation-tier breakdown, so a recruiter gets the gist before
    reading every candidate individually.
    """
    scores = [c.ranking.overall_score for c in candidates]
    recommendation_counts: dict = {}
    for candidate in candidates:
        label = candidate.ranking.recommendation or "Unranked"
        recommendation_counts[label] = recommendation_counts.get(label, 0) + 1
    rec_summary = ", ".join(f"{label}: {count}" for label, count in recommendation_counts.items())

    data = [
        ["Total Candidates", str(len(candidates))],
        ["Highest Score", f"{max(scores):.1f}"],
        ["Average Score", f"{sum(scores) / len(scores):.1f}"],
        ["Lowest Score", f"{min(scores):.1f}"],
        ["Recommendation Breakdown", rec_summary],
    ]
    table = Table([["Executive Summary", ""]] + data, colWidths=[2.2 * inch, 4.1 * inch])
    table.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME["primary"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _build_footer_paragraph(candidates: List[Candidate], body_style) -> Paragraph:
    """Reproducibility footer: scoring version + the weight split actually
    used, so a report printed today can be traced back to how it was scored.
    """
    sample = candidates[0].ranking
    weights = sample.weights_snapshot
    weight_str = "/".join(f"{int(w * 100)}" for w in weights.values()) if weights else "N/A"
    footer_style = ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.grey, spaceBefore=8)
    return Paragraph(
        f"Scoring version: {_esc(sample.scoring_version or 'N/A')} | "
        f"Weights (skills/experience/education/certification/project): {weight_str}",
        footer_style,
    )


def _build_ranking_table(candidates: List[Candidate]) -> Table:
    header = ["Rank", "Name", "Overall Score", "Recommendation", "Confidence"]
    data = [header]
    for candidate in candidates:
        ranking = candidate.ranking
        data.append([
            str(ranking.rank or ""), _esc(candidate.display_name), f"{ranking.overall_score:.1f}",
            _esc(ranking.recommendation or ""), _esc(ranking.confidence_level or ""),
        ])

    table = Table(data, colWidths=[0.6 * inch, 2.0 * inch, 1.2 * inch, 1.6 * inch, 1.0 * inch], repeatRows=1)
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME["primary"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i, candidate in enumerate(candidates, start=1):
        bg = colors.HexColor(f"#{_score_band_hex(candidate.ranking.overall_score)}")
        style_commands.append(("BACKGROUND", (2, i), (2, i), bg))
    table.setStyle(TableStyle(style_commands))
    return table


def _build_candidate_detail(candidate: Candidate, section_style, body_style) -> list:
    ranking = candidate.ranking
    insights = candidate.insights
    elements = [
        Paragraph(f"#{ranking.rank or '—'} {_esc(candidate.display_name)}", section_style),
        Paragraph(
            f"Overall Score: <b>{ranking.overall_score:.1f}/100</b> — {_esc(ranking.recommendation or 'Not ranked')} "
            f"(Confidence: {_esc(ranking.confidence_level or 'N/A')})",
            body_style,
        ),
        Spacer(1, 0.12 * inch),
    ]

    if ranking.category_breakdown:
        breakdown_data = [["Category", "Score", "Weight", "Contribution"]]
        for category, category_score in ranking.category_breakdown.items():
            breakdown_data.append([
                category.replace("_", " ").title(), f"{category_score.raw_score:.0f}",
                f"{category_score.weight * 100:.0f}%", f"{category_score.weighted_contribution:.1f}",
            ])
        breakdown_table = Table(breakdown_data, colWidths=[1.7 * inch, 1.0 * inch, 1.0 * inch, 1.3 * inch])
        breakdown_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME["surface"])),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(breakdown_table)
        elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph("<b>Strengths</b>", body_style))
    for strength in insights.strengths:
        elements.append(Paragraph(f"• {_esc(strength)}", body_style))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph("<b>Weaknesses</b>", body_style))
    for weakness in insights.weaknesses:
        elements.append(Paragraph(f"• {_esc(weakness)}", body_style))
    elements.append(Spacer(1, 0.1 * inch))

    if insights.recruiter_summary:
        elements.append(Paragraph(f"<b>Recruiter Summary:</b> {_esc(insights.recruiter_summary)}", body_style))

    return elements


# --------------------------------------------------------------------------- #
# Export manifest
# --------------------------------------------------------------------------- #
def generate_export_manifest(candidates: List[Candidate], formats_generated: List[str]) -> bytes:
    """Small metadata file describing a multi format export batch (e.g. a
    Module 7 "export all" action generating CSV+Excel+PDF+JSON together) —
    useful for debugging which files came from which run.
    """
    scoring_version = candidates[0].ranking.scoring_version if candidates else None
    manifest = {
        "export_time": datetime.now().isoformat(),
        "candidate_count": len(candidates),
        "report_version": scoring_version,
        "files_generated": list(formats_generated),
    }
    return json.dumps(manifest, indent=2).encode("utf-8")
