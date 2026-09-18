"""
core/extractor.py
==================
Turns cleaned text into structured data:
  - SkillDatabase: scans arbitrary text for any of the ~217 known skills
    (case/spacing/hyphen-insensitive), shared by both extraction paths below.
  - ResumeExtractor.extract(): ParsedDocument -> fully populated Candidate.
  - ResumeExtractor.extract_job_description(): raw JD text -> JobDescription.

Both extraction paths live in the same module because they're the same
underlying task  "find structured facts in unstructured text"  applied to
two different document types, and JD extraction reuses 100% of the skill
detection engine built for resumes.

Every extraction step here is a regex/heuristic, not a language model: it
will not catch every formatting variant, and that's a deliberate, documented
trade-off (see ParsingDiagnostics.warnings on the returned Candidate) rather
than a silent failure  core/ranking.py and the chatbot are both designed to
treat low-confidence extractions with appropriate caution.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from config import CERTIFICATIONS_FILE, DEGREES_FILE, KNOWN_ENTITIES_FILE, SKILLS_FILE
from core.parser import ParsedDocument
from models.candidate import (
    Candidate,
    Education,
    ExperienceEntry,
    ProcessingStatus,
    ProjectEntry,
    ResumeMetadata,
)
from models.job_description import JobDescription
from utils.constants import (
    CURRENT_ROLE_KEYWORDS,
    EMAIL_PATTERN,
    GITHUB_PATTERN,
    LINKEDIN_PATTERN,
    PHONE_PATTERN,
    PORTFOLIO_URL_PATTERN,
    SECTION_HEADERS,
    SENIORITY_KEYWORDS,
    YEAR_RANGE_PATTERN,
    YEARS_OF_EXPERIENCE_PATTERN,
)
from utils.helpers import clean_text, get_logger, normalize_for_matching

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Skill detection engine
# --------------------------------------------------------------------------- #
class SkillDatabase:
    """Loads data/skills.json once and exposes detect() to scan arbitrary
    text for known skills/aliases.

    Matching strategy: build ONE compiled alternation of every skill and
    alias, sorted longest-first (so "TensorFlow" is tried before a
    coincidental partial match, and "C++" is tried before bare "C"), with
    each alternative using flexible `[\\s\\-_./]*` separators between words
    (so "Tensor Flow" / "tensor-flow" / "TensorFlow" all match one pattern)
    and non-alphanumeric lookaround boundaries instead of `\\b` (so "R" or
    "Go" match as standalone tokens but never mid-word, e.g. inside "HR" or
    "Google" — `\\b` alone doesn't reliably prevent this since it treats
    every alnum/non-alnum transition as a boundary regardless of context).

    This is regex-based, not embedding-based, by design — the project
    constraints rule out PyTorch/GPU-backed similarity models, and a
    curated alias list gets most of the practical benefit at near-zero cost.
    """

    def __init__(self, skills_file=SKILLS_FILE):
        self._category_of: Dict[str, str] = {}
        self._reverse_lookup: Dict[str, str] = {}
        self._pattern = self._build(skills_file)

    def _build(self, skills_file) -> re.Pattern:
        raw = json.loads(skills_file.read_text(encoding="utf-8"))
        all_terms: List[str] = []
        for category, entries in raw["categories"].items():
            for entry in entries:
                canonical = entry["name"]
                self._category_of[canonical] = category
                variants = [canonical, *entry.get("aliases", [])]
                for variant in variants:
                    self._reverse_lookup[normalize_for_matching(variant)] = canonical
                    all_terms.append(variant)

        all_terms.sort(key=len, reverse=True)
        pattern_bodies = [self._term_to_pattern_body(t) for t in all_terms]
        combined = "|".join(pattern_bodies)
        return re.compile(rf"(?<![A-Za-z0-9])(?:{combined})(?![A-Za-z0-9])", re.IGNORECASE)

    _CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

    @staticmethod
    def _term_to_pattern_body(term: str) -> str:
        # Split on explicit separators first (space/hyphen/underscore/dot/slash),
        # then further split each piece on camelCase boundaries — otherwise a
        # solid compound word like "TensorFlow" (no separator characters in
        # the source string at all) could never tolerate "Tensor Flow" or
        # "tensor-flow", since there'd be nothing to insert a flexible
        # separator between.
        rough_parts = [p for p in re.split(r"[\s\-_./]+", term.strip()) if p]
        parts: List[str] = []
        for piece in rough_parts:
            parts.extend(p for p in SkillDatabase._CAMEL_BOUNDARY.split(piece) if p)
        escaped = [re.escape(p) for p in parts]
        return r"[\s\-_./]*".join(escaped)

    def detect(self, text: str) -> List[str]:
        """Return the sorted list of unique canonical skill names found in text."""
        if not text:
            return []
        found = set()
        for match in self._pattern.finditer(text):
            canonical = self._reverse_lookup.get(normalize_for_matching(match.group(0)))
            if canonical:
                found.add(canonical)
        return sorted(found)

    def category_of(self, canonical_skill: str) -> Optional[str]:
        return self._category_of.get(canonical_skill)

    @property
    def all_skills(self) -> List[str]:
        return sorted(self._category_of.keys())


@lru_cache(maxsize=1)
def get_skill_database() -> SkillDatabase:
    """Process-wide cached instance — compiling the ~350-alternative regex
    is the expensive part; do it once, not once per resume. Consumers
    should still prefer passing an explicit instance in (see
    ResumeExtractor.__init__) rather than calling this directly, except at
    the top of the dependency graph (app.py).
    """
    return SkillDatabase()


# --------------------------------------------------------------------------- #
# Resume + job description extraction
# --------------------------------------------------------------------------- #
class ResumeExtractor:
    """Extracts structured Candidate/JobDescription data from cleaned text.

    Accepts its SkillDatabase dependency via constructor (DI) so tests can
    inject a small fake skill set instead of the full production database.
    """

    _RESUME_TITLE_WORDS = {"resume", "curriculum vitae", "cv", "biodata", "bio-data"}
    _JD_PREFERRED_MARKERS = ("preferred", "nice to have", "bonus", "good to have")
    _LOCATION_PATTERN = re.compile(
        r"\b([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+){0,2}),\s*([A-Z]{2}|[A-Z][a-zA-Z]+)\b"
    )
    _CGPA_PATTERN = re.compile(r"(?:cgpa|gpa)\s*[:\-]?\s*[\d]+\.?\d*\s*(?:/\s*[\d]+\.?\d*)?", re.IGNORECASE)
    _YEAR_TOKEN_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
    _LEADERSHIP_KEYWORDS = ("lead", "manager", "head", "director", "chief", "principal")

    # Unicode block ranges for cheap SCRIPT detection (not language ID — see
    # ParsingDiagnostics.detected_script docstring for what this can and can't tell us).
    _SCRIPT_RANGES = {
        "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
        "Cyrillic": [(0x0400, 0x04FF)],
        "CJK": [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0x30A0, 0x30FF)],
        "Devanagari": [(0x0900, 0x097F)],
    }

    _CONFIDENCE_WEIGHTS = {
        "full_name": 15, "email": 15, "phone": 10, "skills": 20,
        "education": 15, "experience": 20, "summary": 5,
    }

    def __init__(self, skill_database: Optional[SkillDatabase] = None):
        self.skill_database = skill_database or get_skill_database()
        self._degree_levels, self._degree_fields, self._honors_patterns = self._load_degree_data()
        known_entities = json.loads(KNOWN_ENTITIES_FILE.read_text(encoding="utf-8"))
        self._known_universities: List[str] = known_entities.get("universities", [])
        self._known_languages: List[str] = known_entities.get("spoken_languages", [])
        self._cert_providers: dict = json.loads(CERTIFICATIONS_FILE.read_text(encoding="utf-8"))["providers"]
        self._all_header_strings = {h.lower() for headers in SECTION_HEADERS.values() for h in headers}
        # \b-bounded so "architect" doesn't false-positive inside "architecture",
        # "director" inside "directory", etc. — plain substring checks would.
        self._seniority_patterns: Dict[str, List[re.Pattern]] = {
            level: [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords]
            for level, keywords in SENIORITY_KEYWORDS.items()
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(self, parsed_doc: ParsedDocument) -> Candidate:
        """Turn a ParsedDocument into a fully populated Candidate."""
        start = time.perf_counter()
        candidate = Candidate(
            metadata=ResumeMetadata(
                filename=parsed_doc.filename,
                file_size_bytes=parsed_doc.file_size_bytes,
                page_count=parsed_doc.page_count,
                content_hash=parsed_doc.content_hash,
                parse_time_ms=parsed_doc.parse_time_ms,
            ),
            raw_text=parsed_doc.raw_text,
        )
        candidate.diagnostics.status = ProcessingStatus.PARSED

        text = parsed_doc.cleaned_text
        header_block = self._header_block(text)

        candidate.email = self._extract_email(text)
        candidate.phone = self._extract_phone(header_block)
        candidate.linkedin = self._extract_linkedin(text)
        candidate.github = self._extract_github(text)
        candidate.full_name = self._extract_name(header_block)
        candidate.location = self._extract_location(header_block)

        sections = self._split_into_sections(text)
        candidate.summary = sections.get("summary") or None
        candidate.skills = self.skill_database.detect(text)
        candidate.education = self._extract_education(sections.get("education", ""))
        candidate.experience, candidate.total_years_experience, candidate.seniority_level = (
            self._extract_experience(sections.get("experience", ""))
        )
        candidate.projects = self._extract_projects(sections.get("projects", ""))
        candidate.certifications = self._extract_certifications(sections.get("certifications", ""), text)
        candidate.languages = self._extract_languages(sections.get("languages", ""))
        candidate.achievements = self._split_lines(sections.get("achievements", ""))
        candidate.publications = self._split_lines(sections.get("publications", ""))

        self._add_diagnostics_warnings(candidate)
        candidate.diagnostics.detected_script = self._detect_script(text)
        if candidate.diagnostics.detected_script != "Latin":
            candidate.diagnostics.add_warning(
                f"Resume text appears to be primarily {candidate.diagnostics.detected_script} script — "
                "extraction rules here are English/Latin-script oriented, so accuracy may be reduced."
            )
        candidate.diagnostics.confidence, candidate.diagnostics.field_confidence = self._compute_confidence(candidate)
        candidate.diagnostics.status = ProcessingStatus.EXTRACTED

        elapsed_ms = (time.perf_counter() - start) * 1000
        candidate.metadata.extract_time_ms = elapsed_ms
        candidate.metadata.processing_time_ms = parsed_doc.parse_time_ms + elapsed_ms
        logger.info(
            "Extracted '%s' in %.1fms (confidence=%.0f%%, %d skills, %d warnings)",
            parsed_doc.filename, elapsed_ms, candidate.diagnostics.confidence,
            len(candidate.skills), len(candidate.diagnostics.warnings),
        )
        return candidate

    def extract_job_description(self, raw_text: str) -> JobDescription:
        """Turn raw job description text into a structured JobDescription."""
        cleaned = clean_text(raw_text)
        required_text, preferred_text = self._split_required_preferred(cleaned)

        required_skills = self.skill_database.detect(required_text)
        preferred_skills = [s for s in self.skill_database.detect(preferred_text) if s not in required_skills]
        if not required_skills and not preferred_skills:
            required_skills = self.skill_database.detect(cleaned)

        jd = JobDescription(
            raw_text=raw_text,
            title=self._extract_jd_title(cleaned),
            cleaned_text=cleaned,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_years_experience=self._extract_min_years(cleaned),
            required_education_level=self._extract_required_education_level(cleaned),
            required_certifications=self._extract_certifications("", cleaned),
        )
        logger.info(
            "Extracted JD '%s': %d required skills, %d preferred, min_years=%s",
            jd.title, len(jd.required_skills), len(jd.preferred_skills), jd.min_years_experience,
        )
        return jd

    # ------------------------------------------------------------------ #
    # Contact info
    # ------------------------------------------------------------------ #
    def _header_block(self, text: str, max_lines: int = 12) -> str:
        """Lines before the first recognized section header (capped at
        max_lines) — i.e. just the contact block. Critical for location/phone
        extraction: without this cap, a comma-separated capitalized list
        further down (e.g. "Python, TensorFlow" in a SKILLS section) can look
        exactly like a "City, State" location match.
        """
        lines = text.split("\n")
        limit = min(max_lines, len(lines))
        for i, line in enumerate(lines[:max_lines]):
            key = line.strip().rstrip(":").lower()
            if key in self._all_header_strings:
                limit = i
                break
        return "\n".join(lines[:limit])

    @staticmethod
    def _extract_email(text: str) -> Optional[str]:
        match = EMAIL_PATTERN.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_phone(header_text: str) -> Optional[str]:
        match = PHONE_PATTERN.search(header_text)
        if not match:
            return None
        candidate = match.group(0).strip()
        # Loose pattern by necessity (international formats vary widely) —
        # require enough digits to rule out short numeric noise like a year.
        if sum(ch.isdigit() for ch in candidate) < 7:
            return None
        return candidate

    @staticmethod
    def _extract_linkedin(text: str) -> Optional[str]:
        match = LINKEDIN_PATTERN.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_github(text: str) -> Optional[str]:
        match = GITHUB_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_name(self, header_text: str) -> Optional[str]:
        lines = [line.strip() for line in header_text.split("\n") if line.strip()]
        for line in lines[:6]:
            key = line.rstrip(":").lower()
            if key in self._RESUME_TITLE_WORDS or key in self._all_header_strings:
                continue
            if EMAIL_PATTERN.search(line) or LINKEDIN_PATTERN.search(line) or GITHUB_PATTERN.search(line):
                continue
            if any(ch.isdigit() for ch in line):
                continue
            words = line.split()
            if not (1 <= len(words) <= 5):
                continue
            if not self._looks_like_name(line):
                continue
            return line.title() if line.isupper() else line
        return None

    @staticmethod
    def _looks_like_name(line: str) -> bool:
        """Unicode-aware name-shape check: alphabetic characters (in any
        script — José, Müller, and Arabic-script names should all pass)
        plus space/period/hyphen/apostrophe, capped at a plausible name
        length. Deliberately NOT an ASCII-only `[A-Za-z]` regex — that would
        reject the accented/non-Latin names clean_text() now preserves.
        """
        if not line or len(line) > 45 or not line[0].isalpha():
            return False
        allowed_punctuation = " .-'"
        return all(ch.isalpha() or ch in allowed_punctuation for ch in line)

    def _extract_location(self, header_text: str) -> Optional[str]:
        # Contact lines routinely mix email/phone/location on one line
        # ("jane@x.com | +1 555... | San Francisco, CA") — search the whole
        # line rather than skipping it outright for containing an email.
        for line in header_text.split("\n"):
            match = self._LOCATION_PATTERN.search(line)
            if match:
                return match.group(0).strip()
        return None

    # ------------------------------------------------------------------ #
    # Section splitting — the fix for "every resume uses different headings"
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_into_sections(text: str) -> Dict[str, str]:
        lines = text.split("\n")
        header_lookup = {h.lower(): category for category, headers in SECTION_HEADERS.items() for h in headers}

        boundaries: List[Tuple[int, str]] = []
        for i, line in enumerate(lines):
            key = line.strip().rstrip(":").strip().lower()
            if key in header_lookup:
                boundaries.append((i, header_lookup[key]))

        sections: Dict[str, str] = {}
        for idx, (line_idx, category) in enumerate(boundaries):
            start = line_idx + 1
            end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if not block:
                continue
            sections[category] = f"{sections[category]}\n{block}" if category in sections else block
        return sections

    @staticmethod
    def _split_lines(section_text: str) -> List[str]:
        if not section_text.strip():
            return []
        return [line.strip(" -•\t") for line in section_text.split("\n") if line.strip()]

    # ------------------------------------------------------------------ #
    # Education
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_degree_data():
        data = json.loads(DEGREES_FILE.read_text(encoding="utf-8"))
        levels = [
            {
                "level": entry["level"],
                "rank": entry["rank"],
                # \b-wrapped: short abbreviation patterns like "m\.?e\.?" (M.E.)
                # or "b\.?a\.?" (B.A.) would otherwise match mid-word inside
                # ordinary text — e.g. "require-ME-nts" — without boundaries.
                "patterns": [re.compile(rf"\b(?:{p})\b", re.IGNORECASE) for p in entry["patterns"]],
            }
            for entry in data["degree_levels"]
        ]
        fields = data["common_fields"]
        honors_patterns = [re.compile(rf"\b(?:{p})\b", re.IGNORECASE) for p in data["honors_patterns"]]
        return levels, fields, honors_patterns

    def _extract_education(self, section_text: str) -> List[Education]:
        if not section_text.strip():
            return []
        return [self._parse_education_block(b) for b in self._split_education_blocks(section_text) if b.strip()]

    def _split_education_blocks(self, text: str) -> List[str]:
        paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
        lines = text.strip().split("\n")
        degree_line_idxs = [i for i, line in enumerate(lines) if self._line_mentions_degree(line)]
        if len(degree_line_idxs) <= 1:
            return [text]
        blocks = []
        for idx, start in enumerate(degree_line_idxs):
            end = degree_line_idxs[idx + 1] if idx + 1 < len(degree_line_idxs) else len(lines)
            blocks.append("\n".join(lines[start:end]))
        return blocks

    def _line_mentions_degree(self, line: str) -> bool:
        return any(pattern.search(line) for level in self._degree_levels for pattern in level["patterns"])

    def _parse_education_block(self, block: str) -> Education:
        edu = Education(raw_line=block.strip())

        best_level, best_rank = None, -1
        for level in self._degree_levels:
            if any(p.search(block) for p in level["patterns"]) and level["rank"] > best_rank:
                best_rank, best_level = level["rank"], level["level"]
        edu.degree_level = best_level

        for field_name in self._degree_fields:
            if re.search(rf"\b{re.escape(field_name)}\b", block, re.IGNORECASE):
                edu.field_of_study = field_name
                break

        years = [int(y) for y in self._YEAR_TOKEN_PATTERN.findall(block)]
        if years:
            edu.graduation_year = max(years)

        cgpa_match = self._CGPA_PATTERN.search(block)
        if cgpa_match:
            edu.cgpa = cgpa_match.group(0).strip()

        for pattern in self._honors_patterns:
            match = pattern.search(block)
            if match:
                edu.honors = match.group(0)
                break

        edu.institution = self._guess_institution(block)
        first_line = block.strip().split("\n")[0].strip()
        edu.degree = first_line or None
        return edu

    def _guess_institution(self, block: str) -> Optional[str]:
        lower_block = block.lower()
        for uni in self._known_universities:
            if uni.lower() in lower_block:
                return uni
        for line in block.split("\n"):
            if re.search(r"\b(university|college|institute|school|academy)\b", line, re.IGNORECASE):
                return line.strip()
        return None

    # ------------------------------------------------------------------ #
    # Experience
    # ------------------------------------------------------------------ #
    def _extract_experience(self, section_text: str) -> Tuple[List[ExperienceEntry], float, Optional[str]]:
        if not section_text.strip():
            return [], 0.0, None
        entries = [
            e for b in self._split_experience_blocks(section_text)
            if b.strip() and (e := self._parse_experience_block(b)) is not None
        ]
        # Sort newest-first regardless of the order roles appeared in the
        # document — most resumes are already reverse-chronological, but not
        # all, and downstream consumers (UI, chatbot) shouldn't have to guess.
        entries.sort(key=lambda e: (not e.is_current, -(e.start_year or 0)))
        # NOTE: durations are summed independently per entry. This slightly
        # overestimates total experience if two listed roles genuinely
        # overlapped in time (e.g. a concurrent freelance gig) — a fully
        # correct implementation would merge overlapping date ranges, which
        # we've deliberately scoped out; see Module 2 notes.
        total_years = round(sum(e.duration_years for e in entries), 1)
        seniority = self._infer_seniority(entries, total_years, section_text)
        return entries, total_years, seniority

    @staticmethod
    def _split_experience_blocks(text: str) -> List[str]:
        matches = list(YEAR_RANGE_PATTERN.finditer(text))
        if not matches:
            return [text] if text.strip() else []
        blocks = []
        for i, match in enumerate(matches):
            line_start = text.rfind("\n", 0, match.start()) + 1
            end = text.rfind("\n", 0, matches[i + 1].start()) + 1 if i + 1 < len(matches) else len(text)
            blocks.append(text[line_start:end])
        return blocks

    def _parse_experience_block(self, block: str) -> Optional[ExperienceEntry]:
        match = YEAR_RANGE_PATTERN.search(block)
        if not match:
            return None
        start_year = self._to_int(match.group("start"))
        end_token = match.group("end").lower()
        is_current = end_token in CURRENT_ROLE_KEYWORDS or end_token in ("present", "current")
        end_year = None if is_current else self._to_int(match.group("end"))

        lines = [line.strip() for line in block.split("\n") if line.strip()]
        header_line = lines[0] if lines else ""
        header_clean = YEAR_RANGE_PATTERN.sub("", header_line).strip(" -|,")
        designation, company = self._split_title_company(header_clean)
        responsibilities = [line.lstrip("-• ").strip() for line in lines[1:] if line.strip()]

        combined_lower = header_clean.lower()
        return ExperienceEntry(
            company=company,
            designation=designation,
            start_year=start_year,
            end_year=end_year,
            is_current=is_current,
            duration_text=match.group(0),
            responsibilities=responsibilities,
            is_internship="intern" in combined_lower,
            is_freelance=("freelance" in combined_lower) or ("self-employed" in combined_lower),
            is_leadership=any(k in combined_lower for k in self._LEADERSHIP_KEYWORDS),
            raw_block=block.strip(),
        )

    @staticmethod
    def _to_int(token: str) -> Optional[int]:
        try:
            return int(token)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _split_title_company(header: str) -> Tuple[Optional[str], Optional[str]]:
        for sep in (" at ", " @ ", " - ", " | ", ","):
            if sep in header:
                parts = [p.strip() for p in header.split(sep, 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    return parts[0], parts[1]
        return (header or None), None

    def _infer_seniority(
        self, entries: List[ExperienceEntry], total_years: float, section_text: str
    ) -> Optional[str]:
        """Current-role-keyword first, years-of-experience as fallback — not
        a pure keyword count, per the assignment's "based on resume content
        rather than keyword counting alone" guidance (a true content-aware
        judgment would need an LLM call, out of scope for an extractor that
        must stay deterministic and offline-testable).

        Deliberately checks the CURRENT/most-recent role's title first,
        rather than scanning every historical title equally — otherwise an
        old "Junior Developer" entry from years ago would outrank several
        subsequent years of unmarked-title experience.
        """
        latest = self._current_or_latest_entry(entries)
        if latest and latest.designation:
            level = self._match_seniority_keyword(latest.designation)
            if level:
                return level

        if total_years >= 8:
            return "Lead"
        if total_years >= 5:
            return "Senior"
        if total_years >= 2:
            return "Mid-Level"
        if total_years > 0:
            return "Junior"
        if entries:
            return self._match_seniority_keyword(section_text)
        return None

    def _match_seniority_keyword(self, text: str) -> Optional[str]:
        for level in ("Architect", "Manager", "Lead", "Senior", "Mid-Level", "Junior"):
            if any(pattern.search(text) for pattern in self._seniority_patterns[level]):
                return level
        return None

    @staticmethod
    def _current_or_latest_entry(entries: List[ExperienceEntry]) -> Optional[ExperienceEntry]:
        if not entries:
            return None
        current = [e for e in entries if e.is_current]
        if current:
            return current[0]
        dated = [e for e in entries if e.start_year is not None]
        if dated:
            return max(dated, key=lambda e: e.start_year)
        return entries[0]

    # ------------------------------------------------------------------ #
    # Projects
    # ------------------------------------------------------------------ #
    def _extract_projects(self, section_text: str) -> List[ProjectEntry]:
        if not section_text.strip():
            return []
        blocks = [b for b in re.split(r"\n\s*\n", section_text.strip()) if b.strip()]
        if len(blocks) <= 1:
            blocks = self._split_by_title_lines(section_text)
        return [self._parse_project_block(b) for b in blocks]

    def _split_by_title_lines(self, text: str) -> List[str]:
        lines = text.split("\n")
        title_idxs = [i for i, line in enumerate(lines) if self._looks_like_title(line)]
        if len(title_idxs) <= 1:
            return [text]
        blocks = []
        for idx, start in enumerate(title_idxs):
            end = title_idxs[idx + 1] if idx + 1 < len(title_idxs) else len(lines)
            blocks.append("\n".join(lines[start:end]))
        return blocks

    @staticmethod
    def _looks_like_title(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) >= 70 or stripped.endswith("."):
            return False
        if len(stripped.split()) > 8:
            return False
        if GITHUB_PATTERN.search(stripped) or PORTFOLIO_URL_PATTERN.search(stripped):
            return False  # a bare link is never a project title
        return True

    def _parse_project_block(self, block: str) -> ProjectEntry:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        name = lines[0] if lines else None
        description = "\n".join(lines[1:]).strip()
        github_match = GITHUB_PATTERN.search(block)
        demo_match = PORTFOLIO_URL_PATTERN.search(block)
        demo_link = None
        if demo_match and (not github_match or demo_match.group(0) != github_match.group(0)):
            demo_link = demo_match.group(0)
        return ProjectEntry(
            name=name,
            description=description,
            tech_stack=self.skill_database.detect(block),
            github_link=github_match.group(0) if github_match else None,
            demo_link=demo_link,
            raw_block=block.strip(),
        )

    # ------------------------------------------------------------------ #
    # Certifications & languages
    # ------------------------------------------------------------------ #
    def _extract_certifications(self, section_text: str, full_text: str) -> List[str]:
        found: List[str] = []
        if section_text.strip():
            found.extend(line.strip(" -•\t") for line in section_text.split("\n") if line.strip())

        # Cross-reference provider keywords against the whole document, in
        # case a cert is mentioned inline without a dedicated section — but
        # only add the generic "{Provider} Certification" fallback label if
        # that provider isn't ALREADY represented by an explicit line found
        # above (otherwise "AWS Certified Solutions Architect" plus a
        # generic "AWS Certification" would both show up for the same cert).
        already_mentioned = " ".join(found).lower()
        lower_full = full_text.lower()
        for provider, info in self._cert_providers.items():
            provider_already_covered = (
                provider.lower() in already_mentioned
                or any(keyword in already_mentioned for keyword in info["keywords"])
            )
            if provider_already_covered:
                continue
            if any(keyword in lower_full for keyword in info["keywords"]):
                found.append(f"{provider} Certification")

        seen, deduped = set(), []
        for item in found:
            key = normalize_for_matching(item)
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def _extract_languages(self, section_text: str) -> List[str]:
        if not section_text.strip():
            return []
        tokens = [t.strip() for t in re.split(r"[,\n;•\-]+", section_text) if t.strip()]
        found = []
        for token in tokens:
            for lang in self._known_languages:
                if lang.lower() in token.lower() and lang not in found:
                    found.append(lang)
                    break
        return found

    # ------------------------------------------------------------------ #
    # Job-description-specific helpers
    # ------------------------------------------------------------------ #
    def _split_required_preferred(self, text: str) -> Tuple[str, str]:
        lower = text.lower()
        earliest = min(
            (idx for marker in self._JD_PREFERRED_MARKERS if (idx := lower.find(marker)) != -1),
            default=None,
        )
        if earliest is None:
            return text, ""
        return text[:earliest], text[earliest:]

    @staticmethod
    def _extract_min_years(text: str) -> Optional[float]:
        match = YEARS_OF_EXPERIENCE_PATTERN.search(text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except (ValueError, IndexError, TypeError):
            return None

    def _extract_required_education_level(self, text: str) -> Optional[str]:
        best_level, best_rank = None, -1
        for level in self._degree_levels:
            if any(p.search(text) for p in level["patterns"]) and level["rank"] > best_rank:
                best_rank, best_level = level["rank"], level["level"]
        return best_level

    @staticmethod
    def _extract_jd_title(text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return None
        first = lines[0]
        if len(first) <= 80 and len(first.split()) <= 10:
            return first
        return None

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def _compute_confidence(self, candidate: Candidate) -> Tuple[float, Dict[str, float]]:
        """Each field's OWN confidence is a simple 0/100 presence check,
        deliberately kept unsophisticated — except phone, which gets a
        slightly richer signal since digit-count is a genuinely useful
        proxy for "did we grab a real phone number or noise". The aggregate
        is the same weighted sum as before, now derived from this
        breakdown so there's one source of truth instead of two.
        """
        field_scores = {
            "full_name": 100.0 if candidate.full_name else 0.0,
            "email": 100.0 if candidate.email else 0.0,
            "phone": self._phone_confidence(candidate.phone),
            "skills": 100.0 if candidate.skills else 0.0,
            "education": 100.0 if candidate.education else 0.0,
            "experience": 100.0 if candidate.experience else 0.0,
            "summary": 100.0 if candidate.summary else 0.0,
        }
        overall = sum(
            field_scores[name] * (weight / 100.0) for name, weight in self._CONFIDENCE_WEIGHTS.items()
        )
        return round(overall, 1), {name: round(score, 1) for name, score in field_scores.items()}

    @staticmethod
    def _phone_confidence(phone: Optional[str]) -> float:
        if not phone:
            return 0.0
        digit_count = sum(ch.isdigit() for ch in phone)
        return 100.0 if digit_count >= 10 else 60.0

    def _detect_script(self, text: str) -> str:
        """Cheap Unicode-range heuristic — counts characters by script block
        over a sample and returns whichever is dominant. This is SCRIPT
        detection, not language identification (no library dependency, no
        model): it can tell "this is Arabic-script text" but not "this is
        Urdu vs. Arabic vs. Persian", and every Latin-script language
        (English, French, German, ...) is indistinguishable here.
        """
        sample = text[:2000]
        if not sample.strip():
            return "Unknown"
        script_counts = {name: 0 for name in self._SCRIPT_RANGES}
        latin_count = 0
        for ch in sample:
            if not ch.isalpha():
                continue
            code = ord(ch)
            matched = False
            for name, ranges in self._SCRIPT_RANGES.items():
                if any(lo <= code <= hi for lo, hi in ranges):
                    script_counts[name] += 1
                    matched = True
                    break
            if not matched and ch.isascii():
                latin_count += 1
        dominant_name, dominant_count = max(script_counts.items(), key=lambda kv: kv[1])
        if dominant_count > latin_count and dominant_count >= 20:
            return dominant_name
        return "Latin"

    @staticmethod
    def _add_diagnostics_warnings(candidate: Candidate) -> None:
        if not candidate.full_name:
            candidate.diagnostics.add_warning("Full name could not be reliably detected; using filename instead.")
        if not candidate.email:
            candidate.diagnostics.add_warning("Email address not found.")
        if not candidate.phone:
            candidate.diagnostics.add_warning("Phone number not found.")
        if not candidate.skills:
            candidate.diagnostics.add_warning("No known skills detected in this resume.")
        if not candidate.education:
            candidate.diagnostics.add_warning("Education section could not be parsed or was not found.")
        if not candidate.experience:
            candidate.diagnostics.add_warning("Work experience could not be parsed or was not found.")
