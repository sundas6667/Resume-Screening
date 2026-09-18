"""
utils/constants.py
===================
Regex patterns and static lookup tables shared by the parsing/extraction
layer. Centralized here so a pattern only ever needs to be tuned in one
place, and so core/extractor.py stays free of inline regex literals.
"""
import json
import re

from config import SECTION_ALIASES_FILE

# --------------------------------------------------------------------------- #
# Contact information
# --------------------------------------------------------------------------- #
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s.\-]?)?"          # optional country code
    r"(\(?\d{2,4}\)?[\s.\-]?)"        # area code
    r"\d{3,4}[\s.\-]?\d{3,4}"         # local number
)

LINKEDIN_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_/]+", re.IGNORECASE)
GITHUB_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_/]+", re.IGNORECASE)
PORTFOLIO_URL_PATTERN = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Resume section headers — loaded from data/section_aliases.json rather than
# hardcoded here, since real resumes use wildly inconsistent headings
# ("Experience" vs "Employment History" vs "Career") and this list needs to
# be tunable without a code change as new variants get discovered.
# --------------------------------------------------------------------------- #
SECTION_HEADERS: dict[str, list[str]] = json.loads(
    SECTION_ALIASES_FILE.read_text(encoding="utf-8")
)
SECTION_HEADERS.pop("_note", None)

# Compiled once: matches any known section header at the start of a line (case-insensitive)
_ALL_HEADER_VARIANTS = sorted(
    (h for variants in SECTION_HEADERS.values() for h in variants), key=len, reverse=True
)
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in _ALL_HEADER_VARIANTS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# --------------------------------------------------------------------------- #
# Experience parsing
# --------------------------------------------------------------------------- #
YEAR_RANGE_PATTERN = re.compile(
    r"(?P<start>(?:19|20)\d{2}|present|current)\s*[-–—to]{1,4}\s*(?P<end>(?:19|20)\d{2}|present|current)",
    re.IGNORECASE,
)
YEARS_OF_EXPERIENCE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?", re.IGNORECASE
)
CURRENT_ROLE_KEYWORDS = ("present", "current", "till date", "till now", "ongoing")

SENIORITY_KEYWORDS = {
    "Architect": ["architect", "principal engineer", "distinguished engineer"],
    "Manager": ["engineering manager", "team lead manager", "director", "vp", "head of"],
    "Lead": ["lead", "tech lead", "team lead", "staff engineer"],
    "Senior": ["senior", "sr."],
    "Mid-Level": ["mid-level", "mid level"],
    "Junior": ["junior", "jr.", "entry level", "intern", "trainee", "associate"],
}

# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
GITHUB_LINK_IN_PROJECT_PATTERN = GITHUB_PATTERN
DEMO_LINK_KEYWORDS = ("live demo", "demo:", "deployed at", "hosted at", "live at")

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
BULLET_CHAR_PATTERN = re.compile(r"^[\u2022\u25CF\u25AA\-\*\u2023\u2043]\s*", re.MULTILINE)
# Strips genuine control characters (NUL, form feed, etc.) while preserving
# ALL printable Unicode — the previous version used an ASCII-only allowlist
# ([^\x20-\x7E]) which silently deleted every accented or non-Latin
# character (e.g. "José García" -> "Jos Garca"). \t and \n are kept; \r is
# normalized away separately in clean_text() before this pattern runs.
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
