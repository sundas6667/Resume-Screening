"""
models/candidate.py
====================
Domain model for a candidate and the resume-derived entities that make one
up. These are plain dataclasses with no parsing, scoring, or UI logic 
every other module (extractor, ranking, chatbot, report generator, app)
imports from here rather than passing around loose dicts, so a field rename
only ever has to happen in one place.

Structure: `Candidate` is the aggregate root. Scoring output lives in the
nested `RankingResult`, recruiter-facing narrative lives in `AIInsights`,
and upload/file provenance lives in `ResumeMetadata`  split out so the
"content" fields (name, skills, education...) aren't tangled with the
"derived output" fields that get populated later in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from config import SCHEMA_VERSION
from utils.helpers import generate_id


@dataclass
class Education:
    institution: Optional[str] = None
    degree: Optional[str] = None
    degree_level: Optional[str] = None      # e.g. "Bachelor", "Master" (normalized)
    field_of_study: Optional[str] = None
    graduation_year: Optional[int] = None
    cgpa: Optional[str] = None
    honors: Optional[str] = None
    raw_line: str = ""


@dataclass
class ExperienceEntry:
    company: Optional[str] = None
    designation: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    is_current: bool = False
    duration_text: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    is_internship: bool = False
    is_freelance: bool = False
    is_leadership: bool = False
    raw_block: str = ""

    @property
    def duration_years(self) -> float:
        if self.start_year is None:
            return 0.0
        end = self.end_year if self.end_year and not self.is_current else datetime.now().year
        return max(0.0, float(end - self.start_year))


@dataclass
class ProjectEntry:
    name: Optional[str] = None
    description: str = ""
    tech_stack: List[str] = field(default_factory=list)
    github_link: Optional[str] = None
    demo_link: Optional[str] = None
    raw_block: str = ""

    @property
    def has_deployment_evidence(self) -> bool:
        return bool(self.demo_link or self.github_link)


@dataclass
class ResumeMetadata:
    """File-level provenance  separate from the candidate's *content* so
    the same content_hash logic can dedupe uploads without touching
    anything about who the candidate is.

    Per-stage timings (parse/extract/rank) are kept alongside the total so
    a future Analytics tab can show where the pipeline actually spends its
    time, not just the end-to-end number.
    """
    filename: str = ""
    uploaded_at: datetime = field(default_factory=datetime.now)
    file_size_bytes: int = 0
    page_count: int = 0
    processing_time_ms: float = 0.0   # total, parse + extract + rank
    parse_time_ms: float = 0.0        # PDF bytes -> raw text (core/parser.py)
    extract_time_ms: float = 0.0      # raw text -> structured fields (core/extractor.py)
    rank_time_ms: float = 0.0         # scoring against the job description (core/ranking.py)
    content_hash: Optional[str] = None       # SHA-256 of raw file bytes, for de-dup
    schema_version: str = SCHEMA_VERSION


class ProcessingStatus(str, Enum):
    """Where a candidate currently sits in the pipeline. An enum instead of
    a bare `processed: bool` so the UI can show a meaningful stage (and so a
    partially-failed batch  e.g. one bad PDF among ten  is diagnosable).
    """
    UPLOADED = "Uploaded"
    VALIDATED = "Validated"
    PARSED = "Parsed"
    EXTRACTED = "Extracted"
    RANKED = "Ranked"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class ParsingDiagnostics:
    """Extraction-quality signals, surfaced in the UI so a recruiter knows
    which candidates might need a manual look rather than trusting every
    score equally. `confidence` is a heuristic (e.g. how many expected
    fields were successfully found) computed in core/extractor.py  not a
    model probability, and the UI should label it as such.
    """
    status: ProcessingStatus = ProcessingStatus.UPLOADED
    confidence: Optional[float] = None   # 0-100 heuristic parse-quality estimate
    field_confidence: Dict[str, float] = field(default_factory=dict)  # per-field breakdown
    detected_script: str = "Latin"       # Latin/Arabic/Cyrillic/CJK/Devanagari/Unknown —
    # SCRIPT family only (a cheap Unicode-range heuristic), not true language
    # identification: Latin script covers English/French/German indistinguishably.
    # Exists so a non-Latin resume can be flagged as low-confidence for our
    # English-oriented regex patterns, without pulling in a language-ID library.
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class MatchingResult:
    """Structured output of core/matcher.py  one candidate's match against
    one job description. A single object (rather than scattered fields) so
    ranking, the chatbot, and export/analytics all read the same shape
    instead of each reaching for ad-hoc attributes.

    `matched_skills`/`missing_skills` are the combined required+preferred
    view for simple display; `matched_required`/`missing_required`/etc.
    keep the two apart since missing a REQUIRED skill should weigh very
    differently than missing a PREFERRED one  core/ranking.py needs that
    distinction, and collapsing it here would force Module 4 to
    reverse-engineer which was which.

    `matched_keywords`/`missing_keywords` are a softer signal  significant
    JD phrases found verbatim in the resume that aren't in the skill
    taxonomy (e.g. "regulatory compliance")  still worth surfacing, but
    not conflated with a confirmed skill match.
    """
    similarity_score: float = 0.0  # TF-IDF cosine similarity, scaled 0-100
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)
    matched_required: List[str] = field(default_factory=list)
    matched_preferred: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_preferred: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    explanation: str = ""
    confidence: Optional[float] = None
    confidence_reason: str = ""


@dataclass
class CategoryScore:
    """One scoring category's contribution to the overall score  the audit
    trail unit. `raw_score` is the category's own 0-100 value before
    weighting; `weighted_contribution` is the actual points it added to the
    overall score (raw_score * weight). Keeping raw and weighted separate,
    plus a `reason`, is what makes "why did this candidate get this score"
    answerable from the data instead of requiring a re-derivation.
    """
    raw_score: float = 0.0
    weight: float = 0.0
    weighted_contribution: float = 0.0
    reason: str = ""


@dataclass
class RankingSnapshot:
    """A lightweight record of a PAST ranking result, captured when a
    candidate is re-ranked (e.g. after the recruiter tunes weights in
    Settings and re-runs). Session-scoped, not persisted  this is about
    "compare before/after within this run," not a permanent audit log
    (which would need a real data store, deliberately out of scope).
    """
    overall_score: float
    recommendation: Optional[str]
    scoring_version: str
    weights_snapshot: Dict[str, float] = field(default_factory=dict)
    ranked_at: datetime = field(default_factory=datetime.now)


@dataclass
class RankingResult:
    """Everything the ranking engine produces. Kept separate from the raw
    candidate content so it's obvious at a glance which fields are
    "extracted fact" vs. "computed output"  and so re-ranking against a
    different job description just means replacing this one object.
    """
    skill_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    certification_score: float = 0.0
    project_score: float = 0.0
    semantic_similarity: float = 0.0
    overall_score: float = 0.0
    rank: Optional[int] = None
    confidence_level: Optional[str] = None       # High / Medium / Low
    recommendation: Optional[str] = None          # Highly Recommended / Recommended / Consider / Not Recommended
    # Per-category audit trail — e.g. category_breakdown["skills"].reason
    # explains exactly why the skills category scored what it did.
    category_breakdown: Dict[str, CategoryScore] = field(default_factory=dict)
    # Reproducibility: which scoring logic version and which weight values
    # actually produced this result.
    scoring_version: str = ""
    weights_snapshot: Dict[str, float] = field(default_factory=dict)
    ranked_at: datetime = field(default_factory=datetime.now)


@dataclass
class AIInsights:
    """Recruiter-facing narrative derived from the extracted facts + ranking.
    Anything here must be traceable back to extracted resume content  see
    core/ranking.py, which is the only module allowed to populate this.

    `weaknesses` are factual gaps (what's missing); `improvement_areas` are
    more prescriptive/forward-looking (what would move this candidate up) 
    a useful distinction for a future "suggest interview questions" chatbot
    feature. `recruiter_summary` is a holistic one-liner spanning all five
    scoring categories, distinct from the skills-only `skill_match_explanation`.
    """
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    skill_match_explanation: str = ""
    recruiter_summary: str = ""


@dataclass
class Candidate:
    """A fully parsed candidate profile, progressively enriched as it moves
    through the pipeline: parser -> extractor -> matcher -> ranking engine.
    """
    # Identity / provenance
    candidate_id: str = field(default_factory=lambda: generate_id("cand"))
    metadata: ResumeMetadata = field(default_factory=ResumeMetadata)

    # Contact
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None

    # Content
    summary: Optional[str] = None
    raw_text: str = ""
    skills: List[str] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    experience: List[ExperienceEntry] = field(default_factory=list)
    projects: List[ProjectEntry] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)

    # Derived attributes
    total_years_experience: float = 0.0
    seniority_level: Optional[str] = None  # Junior / Mid-Level / Senior / Lead / Architect / Manager

    # Matching output (populated by core/matcher.py) — see MatchingResult.
    matching: MatchingResult = field(default_factory=MatchingResult)

    # Scoring & narrative (populated by core/ranking.py)
    ranking: RankingResult = field(default_factory=RankingResult)
    insights: AIInsights = field(default_factory=AIInsights)
    ranking_history: List[RankingSnapshot] = field(default_factory=list)

    # Parsing diagnostics — status/confidence/warnings bundled together
    diagnostics: ParsingDiagnostics = field(default_factory=ParsingDiagnostics)

    @property
    def display_name(self) -> str:
        return self.full_name or self.metadata.filename or "Unknown Candidate"

    @property
    def skill_count(self) -> int:
        return len(self.skills)

    @property
    def has_deployed_project(self) -> bool:
        return any(p.has_deployment_evidence for p in self.projects)

    def to_summary_dict(self) -> dict:
        """Compact representation used as chatbot context — deliberately
        excludes raw_text so prompts stay small and the model only ever
        sees structured, already-verified facts (never invented ones).
        """
        return {
            "candidate_id": self.candidate_id,
            "name": self.display_name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "years_experience": self.total_years_experience,
            "seniority_level": self.seniority_level,
            "skills": self.skills,
            "matched_skills": self.matching.matched_skills,
            "missing_skills": self.matching.missing_skills,
            "matched_keywords": self.matching.matched_keywords,
            "match_explanation": self.matching.explanation,
            "certifications": self.certifications,
            "education": [f"{e.degree_level or e.degree} in {e.field_of_study}".strip()
                           for e in self.education if e.degree_level or e.degree],
            "project_count": len(self.projects),
            "overall_score": round(self.ranking.overall_score, 1),
            "skill_score": round(self.ranking.skill_score, 1),
            "experience_score": round(self.ranking.experience_score, 1),
            "education_score": round(self.ranking.education_score, 1),
            "certification_score": round(self.ranking.certification_score, 1),
            "project_score": round(self.ranking.project_score, 1),
            "rank": self.ranking.rank,
            "recommendation": self.ranking.recommendation,
            "strengths": self.insights.strengths,
            "weaknesses": self.insights.weaknesses,
            "parser_confidence": self.diagnostics.confidence,
            "parsing_warnings": self.diagnostics.warnings,
        }

    def to_export_row(self) -> dict:
        """Flat dict = one spreadsheet row, for CSV/Excel export."""
        return {
            "Rank": self.ranking.rank,
            "Name": self.display_name,
            "Email": self.email or "",
            "Phone": self.phone or "",
            "Location": self.location or "",
            "Years Experience": self.total_years_experience,
            "Seniority": self.seniority_level or "",
            "Overall Score": round(self.ranking.overall_score, 1),
            "Skill Score": round(self.ranking.skill_score, 1),
            "Experience Score": round(self.ranking.experience_score, 1),
            "Education Score": round(self.ranking.education_score, 1),
            "Certification Score": round(self.ranking.certification_score, 1),
            "Project Score": round(self.ranking.project_score, 1),
            "Recommendation": self.ranking.recommendation or "",
            "Matched Skills": ", ".join(self.matching.matched_skills),
            "Missing Skills": ", ".join(self.matching.missing_skills),
            "Certifications": ", ".join(self.certifications),
            "Status": self.diagnostics.status.value,
            "Parser Confidence": self.diagnostics.confidence,
            "Source File": self.metadata.filename,
        }

    def to_json(self) -> dict:
        """Full structured serialization for the JSON export option 
        includes everything except raw_text (kept out to avoid bloating
        exports with the entire resume body; the structured fields already
        capture what was extracted from it).
        """
        return {
            "candidate_id": self.candidate_id,
            "schema_version": self.metadata.schema_version,
            "metadata": {
                "filename": self.metadata.filename,
                "uploaded_at": self.metadata.uploaded_at.isoformat(),
                "page_count": self.metadata.page_count,
                "content_hash": self.metadata.content_hash,
                "timing_ms": {
                    "total": self.metadata.processing_time_ms,
                    "parse": self.metadata.parse_time_ms,
                    "extract": self.metadata.extract_time_ms,
                    "rank": self.metadata.rank_time_ms,
                },
            },
            "contact": {
                "full_name": self.full_name, "email": self.email, "phone": self.phone,
                "linkedin": self.linkedin, "github": self.github, "location": self.location,
            },
            "summary": self.summary,
            "skills": self.skills,
            "education": [e.__dict__ for e in self.education],
            "experience": [
                {k: v for k, v in e.__dict__.items() if k != "raw_block"} for e in self.experience
            ],
            "projects": [
                {k: v for k, v in p.__dict__.items() if k != "raw_block"} for p in self.projects
            ],
            "certifications": self.certifications,
            "total_years_experience": self.total_years_experience,
            "seniority_level": self.seniority_level,
            "matching": self.matching.__dict__,
            "ranking": {
                **{k: v for k, v in self.ranking.__dict__.items() if k not in ("category_breakdown", "ranked_at")},
                "category_breakdown": {
                    category: score.__dict__ for category, score in self.ranking.category_breakdown.items()
                },
                "ranked_at": self.ranking.ranked_at.isoformat(),
            },
            "ranking_history": [
                {**snap.__dict__, "ranked_at": snap.ranked_at.isoformat()} for snap in self.ranking_history
            ],
            "insights": self.insights.__dict__,
            "diagnostics": {
                "status": self.diagnostics.status.value,
                "confidence": self.diagnostics.confidence,
                "field_confidence": self.diagnostics.field_confidence,
                "detected_script": self.diagnostics.detected_script,
                "warnings": self.diagnostics.warnings,
            },
        }
