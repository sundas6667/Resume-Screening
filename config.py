"""
config.py
=========
Single source of truth for every configurable value in the application:
file paths, theme tokens, scoring weights, thresholds, and model settings.

Nothing outside this module should hardcode a color, a weight, a path, or a
model name. Scoring weights are persisted to `data/scoring_weights.json` so
they can be tuned from the Settings page at runtime without touching code.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"
LOGS_DIR = BASE_DIR / "logs"
SAMPLE_RESUMES_DIR = BASE_DIR / "sample_resumes"

SKILLS_FILE = DATA_DIR / "skills.json"
DEGREES_FILE = DATA_DIR / "degrees.json"
CERTIFICATIONS_FILE = DATA_DIR / "certifications.json"
WEIGHTS_FILE = DATA_DIR / "scoring_weights.json"
SECTION_ALIASES_FILE = DATA_DIR / "section_aliases.json"
KNOWN_ENTITIES_FILE = DATA_DIR / "known_entities.json"

for _dir in (DATA_DIR, EXPORTS_DIR, LOGS_DIR, SAMPLE_RESUMES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# App metadata
# --------------------------------------------------------------------------- #
APP_NAME = "TalentLens AI"
APP_TAGLINE = "AI-Powered Resume Screening & Candidate Ranking"
APP_VERSION = "1.0.0"
DEVELOPER = "Teerop Technologies — ML & AI Internship"

# Bumped independently of APP_VERSION: MODEL_VERSION changes when scoring/
# extraction *logic* changes (so exported reports can record what produced
# them); SCHEMA_VERSION changes when the Candidate/JobDescription field
# structure changes (so old exported JSON can be recognized as stale).
MODEL_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# File upload constraints
# --------------------------------------------------------------------------- #
ALLOWED_RESUME_EXTENSIONS = (".pdf",)
MAX_FILE_SIZE_MB = 10
MAX_RESUMES_PER_BATCH = 25

# --------------------------------------------------------------------------- #
# Score bands (used for color-coding across the UI)
# --------------------------------------------------------------------------- #
SCORE_THRESHOLD_EXCELLENT = 80  # >= this -> green / "Highly Recommended"
SCORE_THRESHOLD_GOOD = 60       # >= this -> yellow / "Recommended"
# below SCORE_THRESHOLD_GOOD -> red / "Consider" or "Not Recommended"
SCORE_THRESHOLD_MINIMUM = 40    # below this -> "Not Recommended"

# Semantic-similarity bands (TF-IDF cosine similarity, 0-100 scale) — used to
# label a raw similarity number as Low/Medium/High for display and for the
# match-confidence reasoning, kept separate from the overall-score bands
# above since a good candidate can have low text-similarity but strong
# explicit skill matches (or vice versa).
SIMILARITY_THRESHOLD_HIGH = 40
SIMILARITY_THRESHOLD_MEDIUM = 15
# below SIMILARITY_THRESHOLD_MEDIUM -> "Low"

# Version tag for the similarity engine itself (not app/schema version) — if
# tokenization or vectorizer settings change later, historical scores can be
# traced back to which engine produced them.
SIMILARITY_ENGINE_VERSION = "tfidf_v1"

# Version tag for the ranking/scoring logic itself — bumped when the
# category-scoring formulas change (not when just the weight VALUES change;
# a weights_snapshot on each RankingResult captures those). Lets a
# historical score be reproduced: same version + same weights_snapshot +
# same input data = same output.
SCORING_ENGINE_VERSION = "ranking_v1"

# Version tag for the chatbot's system prompt — bumped when the persona/
# rules text in core/chat/prompt_builder.py changes, so a logged answer can
# be traced back to which prompt wording produced it.
PROMPT_VERSION = "1.0"

# Recommendation bands: (minimum overall score, label), checked highest-first.
# Four tiers — matching the original spec's label set exactly — rather than
# finer-grained bands, since the assignment's own color-coding is only three
# bands (green >=80 / yellow 60-79 / red <60); a label can be added later by
# inserting one more (threshold, label) tuple here, no logic changes needed.
RECOMMENDATION_BANDS = [
    (SCORE_THRESHOLD_EXCELLENT, "Highly Recommended"),
    (SCORE_THRESHOLD_GOOD, "Recommended"),
    (SCORE_THRESHOLD_MINIMUM, "Consider"),
    (0, "Not Recommended"),
]

# The assignment's scoring spec lists "Skill Match" (matched/required ratio)
# and "Text Similarity" (TF-IDF cosine) as two formulas under one "Skills"
# category — not two separate weighted categories (the five ScoringWeights
# above already sum to 1.0 on their own). So core/ranking.py blends them
# into a single skill-category score using these weights, rather than
# adding semantic similarity as an independent 6th weighted category.
SKILL_SCORE_KEYWORD_WEIGHT = 0.7
SKILL_SCORE_SEMANTIC_WEIGHT = 0.3

# Confidence-level bands (0-100 -> High/Medium/Low label), shared thinking
# with core/matcher.py's own confidence-reason thresholds for consistency.
CONFIDENCE_THRESHOLD_HIGH = 75
CONFIDENCE_THRESHOLD_MEDIUM = 50

# --------------------------------------------------------------------------- #
# Groq / LLM settings
# --------------------------------------------------------------------------- #
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0.5
GROQ_MAX_TOKENS = 1024
GROQ_TIMEOUT_SECONDS = 30
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"  # optional env var fallback for the sidebar field

# --------------------------------------------------------------------------- #
# Theme tokens (light professional teal dashboard)
# Full 50-900 scale so any component can pick the right shade instead of
# reaching for a single hardcoded hex value.
# --------------------------------------------------------------------------- #
COLOR_TOKENS: Dict[str, Dict[str, str]] = {
    "primary": {
        "50": "#F0FDFA", "100": "#CCFBF1", "200": "#99F6E4", "300": "#5EEAD4",
        "400": "#2DD4BF", "500": "#0F766E", "600": "#0D9488", "700": "#115E59",
        "800": "#134E4A", "900": "#042F2E",
    },
    "secondary": {
        "50": "#F0FDFA", "100": "#CCFBF1", "200": "#99F6E4", "300": "#5EEAD4",
        "400": "#2DD4BF", "500": "#14B8A6", "600": "#0D9488", "700": "#0F766E",
        "800": "#115E59", "900": "#134E4A",
    },
    "success": {
        "50": "#F0FDF4", "300": "#86EFAC", "500": "#22C55E", "700": "#15803D", "900": "#14532D",
    },
    "warning": {
        "50": "#FFFBEB", "300": "#FCD34D", "500": "#F59E0B", "700": "#B45309", "900": "#78350F",
    },
    "danger": {
        "50": "#FEF2F2", "300": "#FCA5A5", "500": "#EF4444", "700": "#B91C1C", "900": "#7F1D1D",
    },
    "neutral": {
        "0": "#FFFFFF", "50": "#F8FAFC", "100": "#F1F5F9", "200": "#E2E8F0",
        "300": "#CBD5E1", "400": "#94A3B8", "500": "#64748B",
        "600": "#475569", "700": "#334155", "800": "#1E293B", "900": "#0F172A",
    },
}

# Semantic aliases so UI code reads intent, not raw hex
THEME = {
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "surface_alt": "#F1F5F9",
    "border": "#E2E8F0",
    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#64748B",
    "primary": "#0F766E",
    "accent": "#14B8A6",
    "success": COLOR_TOKENS["success"]["500"],
    "warning": COLOR_TOKENS["warning"]["500"],
    "danger": COLOR_TOKENS["danger"]["500"],
}

# Spacing scale (px) — every margin/padding in custom CSS should reference these
SPACING = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 20, "2xl": 24, "3xl": 32, "4xl": 40, "5xl": 48, "6xl": 64}

# Border radius scale (px)
RADIUS = {"sm": 6, "md": 10, "lg": 16, "xl": 20, "2xl": 24, "full": 999}

# Shadow scale (CSS box-shadow values)
SHADOWS = {
    "sm": "0 1px 2px rgba(15,23,42,0.06)",
    "md": "0 4px 12px rgba(15,23,42,0.08)",
    "lg": "0 10px 24px rgba(15,23,42,0.10)",
    "xl": "0 20px 40px rgba(15,23,42,0.12)",
}

FONT_FAMILY = "'Inter', 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif"

# Typography scale — every heading/body element in custom CSS should pick a
# token here rather than a one-off font-size/weight combination.
TYPOGRAPHY = {
    "heading_xl": {"size": "32px", "weight": 700, "line_height": "1.2"},
    "heading_lg": {"size": "24px", "weight": 700, "line_height": "1.3"},
    "heading_md": {"size": "18px", "weight": 600, "line_height": "1.4"},
    "body_lg": {"size": "16px", "weight": 400, "line_height": "1.6"},
    "body_md": {"size": "14px", "weight": 400, "line_height": "1.6"},
    "caption": {"size": "12px", "weight": 500, "line_height": "1.4"},
}

TRANSITION_DURATION_MS = {"fast": 150, "base": 250, "slow": 400}


# --------------------------------------------------------------------------- #
# Scoring weights — configurable via file, not hardcoded in ranking logic
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoringWeights:
    """Weighted contribution of each evaluation category to the overall score.

    Values must be in [0, 1] and sum to 1.0 (validated on load/save so a bad
    edit in the Settings page can never silently corrupt rankings).
    """
    skills: float = 0.35
    experience: float = 0.30
    education: float = 0.15
    certification: float = 0.10
    project: float = 0.10

    def validate(self) -> None:
        total = sum(getattr(self, f.name) for f in fields(self))
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Scoring weights must sum to 1.0 (got {total:.3f}). "
                f"Fix data/scoring_weights.json or the Settings page."
            )
        for f in fields(self):
            value = getattr(self, f.name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Weight '{f.name}' must be between 0 and 1 (got {value}).")

    @classmethod
    def load(cls, path: Path = WEIGHTS_FILE) -> "ScoringWeights":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            weights = cls(**raw)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            weights = cls()  # fall back to safe defaults
        weights.validate()
        return weights

    def save(self, path: Path = WEIGHTS_FILE) -> None:
        self.validate()
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def as_percentages(self) -> Dict[str, int]:
        return {f.name: round(getattr(self, f.name) * 100) for f in fields(self)}


# --------------------------------------------------------------------------- #
# Domain-specific config blocks — grouped by concern so a future module never
# has to invent a magic number inline. Kept as dataclasses within this single
# file rather than a config/ package: at current size (~350 lines) one
# well-sectioned file is easier to navigate than a dozen tiny ones. Revisit
# if this file crosses ~500 lines.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerformanceConfig:
    max_uploads_per_batch: int = MAX_RESUMES_PER_BATCH
    max_pdf_size_mb: int = MAX_FILE_SIZE_MB
    cache_enabled: bool = True
    target_seconds_per_10_resumes: float = 10.0
    tfidf_max_features: int = 5000


@dataclass(frozen=True)
class SecurityConfig:
    hash_algorithm: str = "sha256"
    dedupe_by_content_hash: bool = True
    mask_api_key_in_logs: bool = True
    max_text_length_chars: int = 200_000


@dataclass(frozen=True)
class ChartConfig:
    default_height_px: int = 380
    color_sequence: tuple = (
        THEME["primary"], THEME["success"], THEME["warning"],
        THEME["danger"], COLOR_TOKENS["primary"]["300"], COLOR_TOKENS["neutral"]["400"],
    )
    animation_duration_ms: int = 400
    show_hover_tooltips: bool = True


@dataclass(frozen=True)
class ExportConfig:
    formats: tuple = ("csv", "xlsx", "pdf", "json")
    include_charts_in_pdf: bool = True
    company_branding_name: str = APP_NAME


@dataclass(frozen=True)
class AnimationConfig:
    enabled: bool = True
    fade_ms: int = TRANSITION_DURATION_MS["base"]
    slide_ms: int = TRANSITION_DURATION_MS["slow"]
    count_up_ms: int = 800


def get_groq_api_key() -> str:
    """Read a Groq key from the environment as a fallback to the sidebar field."""
    return os.environ.get(GROQ_API_KEY_ENV_VAR, "")
