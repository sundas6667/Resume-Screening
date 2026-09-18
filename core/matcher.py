"""
core/matcher.py
================
Compares a Candidate against a JobDescription and produces a MatchingResult:
matched/missing/extra skills, softer keyword overlap for JD terms outside
the skill taxonomy, and a recruiter-friendly explanation.

This module answers "how well does this resume match, and why"  it does
NOT decide the final ranking score. That combination (skills + experience +
education + certifications + projects, weighted) is core/ranking.py's job;
this module only produces one of its five inputs, plus the semantic
similarity number handed in from core/vector_store.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import SIMILARITY_THRESHOLD_HIGH, SIMILARITY_THRESHOLD_MEDIUM
from core.extractor import SkillDatabase
from core.vector_store import VectorStore, VectorStoreError
from models.candidate import Candidate, MatchingResult
from models.job_description import JobDescription
from utils.helpers import get_logger

logger = get_logger(__name__)

_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")  # 4+ letter words, keeps hyphenated terms whole
_MAX_KEYWORDS_CONSIDERED = 15
_MAX_KEYWORDS_REPORTED = 10


class SkillMatcher:
    """Stateless comparator — safe to share a single instance across an
    entire batch of candidates (no per-candidate mutable state).
    """

    def match(
        self,
        candidate: Candidate,
        job_description: JobDescription,
        similarity_score: float = 0.0,
    ) -> MatchingResult:
        """Compare one candidate against one job description."""
        required = set(job_description.required_skills)
        preferred = set(job_description.preferred_skills)
        relevant = required | preferred
        candidate_skills = set(candidate.skills)

        matched = sorted(candidate_skills & relevant)
        missing = sorted(relevant - candidate_skills)
        extra = sorted(candidate_skills - relevant)

        matched_keywords, missing_keywords = self._match_keywords(candidate, job_description)
        explanation = self._build_explanation(matched, missing, extra, relevant)
        confidence = self._compute_match_confidence(candidate, job_description)
        confidence_reason = self._confidence_reason(candidate, job_description, confidence)

        result = MatchingResult(
            similarity_score=similarity_score,
            matched_skills=matched,
            missing_skills=missing,
            extra_skills=extra,
            matched_required=sorted(candidate_skills & required),
            matched_preferred=sorted(candidate_skills & preferred),
            missing_required=sorted(required - candidate_skills),
            missing_preferred=sorted(preferred - candidate_skills),
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            explanation=explanation,
            confidence=confidence,
            confidence_reason=confidence_reason,
        )
        logger.info(
            "Matched '%s': %d/%d relevant skills (%d required missing), similarity=%.1f, confidence=%s",
            candidate.display_name, len(matched), len(relevant) or 1,
            len(result.missing_required), similarity_score, confidence,
        )
        return result

    # ------------------------------------------------------------------ #
    # Keyword overlap (soft signal, outside the skill taxonomy)
    # ------------------------------------------------------------------ #
    def _match_keywords(self, candidate: Candidate, jd: JobDescription) -> tuple[List[str], List[str]]:
        """Significant JD words that aren't part of any matched/missing
        skill name — e.g. "regulatory compliance" isn't in the skill
        taxonomy but literal presence in the resume is still a useful signal.
        """
        skill_words = set()
        for skill in (*jd.required_skills, *jd.preferred_skills):
            skill_words.update(w.lower() for w in _WORD_PATTERN.findall(skill))

        candidate_text_lower = candidate.raw_text.lower()
        seen: List[str] = []
        for word in _WORD_PATTERN.findall(jd.cleaned_text):
            lower = word.lower()
            if lower in ENGLISH_STOP_WORDS or lower in skill_words or lower in seen:
                continue
            seen.append(lower)
            if len(seen) >= _MAX_KEYWORDS_CONSIDERED:
                break

        matched = [w for w in seen if w in candidate_text_lower][:_MAX_KEYWORDS_REPORTED]
        missing = [w for w in seen if w not in candidate_text_lower][:_MAX_KEYWORDS_REPORTED]
        return matched, missing

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_explanation(matched: List[str], missing: List[str], extra: List[str], relevant: set) -> str:
        if not relevant:
            return "No specific skill requirements were extracted from the job description to compare against."

        match_pct = round(len(matched) / len(relevant) * 100) if relevant else 0
        parts: List[str] = []

        if matched:
            shown = matched[:5]
            clause = f"Candidate demonstrates {', '.join(shown)}"
            if len(matched) > 5:
                clause += f", and {len(matched) - 5} more matched skill(s)"
            parts.append(clause)
            if missing:
                shown = missing[:5]
                clause = f"but is missing {', '.join(shown)}"
                if len(missing) > 5:
                    clause += f" and {len(missing) - 5} other required/preferred skill(s)"
                parts.append(clause)
        elif missing:
            shown = missing[:5]
            clause = f"Candidate shows none of the required or preferred skills for this role — missing {', '.join(shown)}"
            if len(missing) > 5:
                clause += f" and {len(missing) - 5} other(s)"
            parts.append(clause)
        else:
            parts.append("Candidate shows none of the required or preferred skills for this role")

        sentence = " ".join(parts).strip()
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        sentence += f" Overall skill match: {match_pct}%."
        if extra:
            sentence += f" Also brings {len(extra)} additional skill(s) beyond the stated requirements."
        return sentence

    # ------------------------------------------------------------------ #
    # Confidence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_match_confidence(candidate: Candidate, jd: JobDescription) -> float:
        """How much to trust this match  ties back to the candidate's own
        parsing confidence (garbage extraction in means an unreliable match
        out) and penalizes JDs where skill extraction found nothing to
        compare against in the first place.
        """
        if not jd.required_skills and not jd.preferred_skills:
            return 30.0
        base = candidate.diagnostics.confidence
        return round(base, 1) if base is not None else 50.0

    @staticmethod
    def _confidence_reason(candidate: Candidate, jd: JobDescription, confidence: float) -> str:
        """Short, human-readable justification for the confidence number —
        so a recruiter (or the chatbot) can see WHY a match is trusted or
        not, not just the bare percentage.
        """
        if not jd.required_skills and not jd.preferred_skills:
            return "Low confidence: no skill requirements could be extracted from the job description."

        reasons = []
        if candidate.diagnostics.confidence is not None:
            reasons.append(f"resume parsing confidence was {candidate.diagnostics.confidence:.0f}%")
        if candidate.diagnostics.warnings:
            reasons.append(f"{len(candidate.diagnostics.warnings)} parsing warning(s) were raised")
        else:
            reasons.append("no parsing warnings were raised")

        level = "High" if confidence >= 75 else "Medium" if confidence >= 50 else "Low"
        return f"{level} confidence: " + "; ".join(reasons) + "."


# --------------------------------------------------------------------------- #
# Batch orchestration
# --------------------------------------------------------------------------- #
def match_candidates(
    candidates: List[Candidate],
    job_description: JobDescription,
    vector_store: Optional[VectorStore] = None,
    matcher: Optional[SkillMatcher] = None,
) -> List[Candidate]:
    """Convenience orchestrator for the common case: build/reuse a
    VectorStore for semantic similarity, then run skill matching for every
    candidate, populating `candidate.matching` in place.

    If the vector store fails to build (e.g. all empty text  see
    VectorStoreError), matching still proceeds with similarity_score=0.0
    rather than failing the whole batch; skill/keyword matching don't
    depend on it.

    Returns the same list, for chaining.
    """
    matcher = matcher or SkillMatcher()
    similarity_scores: Dict[str, float] = {}

    if vector_store is not None:
        candidate_texts = {c.candidate_id: c.raw_text for c in candidates if c.raw_text.strip()}
        if candidate_texts:
            try:
                vector_store.build(job_description.cleaned_text, candidate_texts)
                similarity_scores = vector_store.similarity_scores()
            except VectorStoreError as exc:
                logger.warning("Vector store unavailable, proceeding without semantic similarity: %s", exc)

    for candidate in candidates:
        similarity = similarity_scores.get(candidate.candidate_id, 0.0)
        candidate.matching = matcher.match(candidate, job_description, similarity_score=similarity)

    return candidates


# --------------------------------------------------------------------------- #
# Skill categorization (derived from SkillDatabase, not hardcoded fields —
# category names live in one place, data/skills.json, so this stays correct
# if the taxonomy changes instead of drifting out of sync)
# --------------------------------------------------------------------------- #
def categorize_skills(skills: List[str], skill_database: SkillDatabase) -> Dict[str, List[str]]:
    """Group a flat skill list by SkillDatabase category, e.g. for an
    analytics view showing "matched: 3 Programming Languages, 2 Cloud &
    DevOps, 1 AI/ML" instead of one undifferentiated list.
    """
    grouped: Dict[str, List[str]] = {}
    for skill in skills:
        category = skill_database.category_of(skill) or "Other"
        grouped.setdefault(category, []).append(skill)
    return grouped


def categorize_similarity(score: float) -> str:
    """Label a raw 0-100 similarity score as Low/Medium/High using the
    configured thresholds  kept separate from the overall-score bands in
    config.py since a candidate can have strong explicit skill matches with
    comparatively low free-text similarity, or vice versa.
    """
    if score >= SIMILARITY_THRESHOLD_HIGH:
        return "High"
    if score >= SIMILARITY_THRESHOLD_MEDIUM:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# Batch statistics
# --------------------------------------------------------------------------- #
@dataclass
class BatchMatchStatistics:
    """Aggregate view across a whole batch of matched candidates — the
    numbers a dashboard's summary row needs, computed once here instead of
    recomputed ad hoc in the UI layer.
    """
    candidate_count: int = 0
    average_similarity: float = 0.0
    highest_similarity: float = 0.0
    lowest_similarity: float = 0.0
    average_skill_coverage: float = 0.0  # mean % of (required+preferred) skills matched
    vocabulary_size: Optional[int] = None


def compute_batch_statistics(
    candidates: List[Candidate],
    job_description: JobDescription,
    vector_store: Optional[VectorStore] = None,
) -> BatchMatchStatistics:
    """Summarize match quality across a batch. Safe to call on an empty
    list or on candidates that haven't been matched yet (all zero stats).
    """
    if not candidates:
        return BatchMatchStatistics()

    similarities = [c.matching.similarity_score for c in candidates]
    relevant_count = len(set(job_description.required_skills) | set(job_description.preferred_skills))
    coverages = [
        (len(c.matching.matched_skills) / relevant_count * 100) if relevant_count else 0.0
        for c in candidates
    ]

    return BatchMatchStatistics(
        candidate_count=len(candidates),
        average_similarity=round(sum(similarities) / len(similarities), 1),
        highest_similarity=round(max(similarities), 1),
        lowest_similarity=round(min(similarities), 1),
        average_skill_coverage=round(sum(coverages) / len(coverages), 1),
        vocabulary_size=vector_store.vocabulary_size() if vector_store is not None and vector_store.is_built else None,
    )
