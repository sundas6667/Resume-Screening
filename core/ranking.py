"""
core/ranking.py
================
Combines everything Modules 2-3 produced into a final weighted score per
candidate: Raw category scores -> Normalization -> Weighted contribution ->
Overall score -> Recommendation. Each stage is a separate, named step (see
RankingEngine._score_* methods and CategoryScore) so "why did this candidate
get 74/100" is answerable directly from candidate.ranking.category_breakdown
rather than requiring a re-derivation.

This module does NOT do any text extraction or skill matching itself , it
consumes candidate.skills/education/experience/projects/certifications
(Module 2) and candidate.matching (Module 3) as already-computed facts.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    CERTIFICATIONS_FILE,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_MEDIUM,
    DEGREES_FILE,
    RECOMMENDATION_BANDS,
    SCORING_ENGINE_VERSION,
    SKILL_SCORE_KEYWORD_WEIGHT,
    SKILL_SCORE_SEMANTIC_WEIGHT,
    ScoringWeights,
)
from models.candidate import AIInsights, Candidate, CategoryScore, ProcessingStatus, RankingResult, RankingSnapshot
from models.job_description import JobDescription
from utils.helpers import clamp, get_logger, normalize_for_matching, safe_divide

logger = get_logger(__name__)

_MIN_PROJECT_DESCRIPTION_WORDS = 8
_MAX_PROJECT_COUNT_BONUS_AT = 3


class RankingEngine:
    """Computes RankingResult for candidates already processed by Modules
    2-3. Accepts its ScoringWeights via constructor injection (DI) rather
    than importing the global config directly, so a Settings page override
    or a test can supply different weights without monkeypatching a module.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights.load()
        self.weights.validate()
        self._degree_ranks = self._load_degree_ranks()
        self._cert_provider_weights = self._load_cert_provider_weights()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def rank(self, candidate: Candidate, job_description: JobDescription) -> RankingResult:
        """Score one candidate against one job description. Assumes
        candidate.matching has already been populated (see core/matcher.py)
        , this method does not compute skill matching itself, only combines
        it with the other four categories.
        """
        start = time.perf_counter()

        skill_score, skill_reason = self._score_skills(candidate, job_description)
        experience_score, experience_reason = self._score_experience(candidate, job_description)
        education_score, education_reason = self._score_education(candidate, job_description)
        certification_score, certification_reason = self._score_certifications(candidate, job_description)
        project_score, project_reason = self._score_projects(candidate, job_description)

        # --- Normalization: every _score_* method already clamps to 0-100,
        # this is a defensive second pass so a future scorer can't silently
        # push the overall score out of range if it forgets to clamp. ---
        raw_scores = {
            "skills": clamp(skill_score),
            "experience": clamp(experience_score),
            "education": clamp(education_score),
            "certification": clamp(certification_score),
            "project": clamp(project_score),
        }
        reasons = {
            "skills": skill_reason, "experience": experience_reason, "education": education_reason,
            "certification": certification_reason, "project": project_reason,
        }

        # --- Weighted contribution per category ---
        weight_map = {
            "skills": self.weights.skills, "experience": self.weights.experience,
            "education": self.weights.education, "certification": self.weights.certification,
            "project": self.weights.project,
        }
        breakdown: Dict[str, CategoryScore] = {}
        overall_score = 0.0
        for category, raw in raw_scores.items():
            weight = weight_map[category]
            contribution = raw * weight
            overall_score += contribution
            breakdown[category] = CategoryScore(
                raw_score=round(raw, 1), weight=weight,
                weighted_contribution=round(contribution, 1), reason=reasons[category],
            )
        overall_score = clamp(round(overall_score, 1))

        result = RankingResult(
            skill_score=raw_scores["skills"],
            experience_score=raw_scores["experience"],
            education_score=raw_scores["education"],
            certification_score=raw_scores["certification"],
            project_score=raw_scores["project"],
            semantic_similarity=round(candidate.matching.similarity_score, 1),
            overall_score=overall_score,
            confidence_level=self._categorize_confidence(candidate.matching.confidence),
            recommendation=self._get_recommendation(overall_score),
            category_breakdown=breakdown,
            scoring_version=SCORING_ENGINE_VERSION,
            weights_snapshot=dict(weight_map),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        candidate.metadata.rank_time_ms = elapsed_ms
        candidate.metadata.processing_time_ms = (
            candidate.metadata.parse_time_ms + candidate.metadata.extract_time_ms + elapsed_ms
        )
        logger.info(
            "Ranked '%s': overall=%.1f (%s) in %.1fms",
            candidate.display_name, overall_score, result.recommendation, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Category scorers — each returns (raw_score_0_to_100, reason_string)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_skills(candidate: Candidate, jd: JobDescription) -> Tuple[float, str]:
        """(Matched / Required) x 100 — the assignment's literal formula —
        blended with the TF-IDF semantic similarity from core/vector_store.py
        (see SKILL_SCORE_*_WEIGHT in config.py for why these are blended
        into one category instead of two independent weighted categories),
        plus a small capped bonus for matched preferred (non-required) skills.
        """
        matching = candidate.matching
        required_count = len(jd.required_skills)
        preferred_count = len(jd.preferred_skills)

        if required_count > 0:
            keyword_score = safe_divide(len(matching.matched_required), required_count) * 100
        elif preferred_count > 0:
            keyword_score = safe_divide(len(matching.matched_preferred), preferred_count) * 100
        else:
            keyword_score = 0.0

        preferred_bonus = 0.0
        if preferred_count > 0 and required_count > 0:  # already used preferred as the base otherwise
            preferred_bonus = safe_divide(len(matching.matched_preferred), preferred_count) * 10

        blended = (
            keyword_score * SKILL_SCORE_KEYWORD_WEIGHT
            + matching.similarity_score * SKILL_SCORE_SEMANTIC_WEIGHT
        )
        score = clamp(blended + preferred_bonus)

        if required_count == 0 and preferred_count == 0:
            reason = "No skill requirements were extracted from the job description to score against."
        else:
            reason = (
                f"Matched {len(matching.matched_required)}/{required_count} required and "
                f"{len(matching.matched_preferred)}/{preferred_count} preferred skills "
                f"(keyword match {keyword_score:.0f}%, semantic similarity {matching.similarity_score:.0f}%)."
            )
        return score, reason

    @staticmethod
    def _score_experience(candidate: Candidate, jd: JobDescription) -> Tuple[float, str]:
        years = candidate.total_years_experience
        if jd.min_years_experience:
            ratio = safe_divide(years, jd.min_years_experience)
            score = clamp(ratio * 100)
            reason = (
                f"{years:.1f} years of experience vs. {jd.min_years_experience:.0f} required "
                f"({score:.0f}% of requirement met)."
            )
        else:
            # No minimum stated in the JD — score against a generic 10-year
            # reference scale rather than leaving experience unscored.
            score = clamp(safe_divide(years, 10) * 100)
            reason = f"{years:.1f} years of experience (JD did not specify a minimum)."
        return score, reason

    def _score_education(self, candidate: Candidate, jd: JobDescription) -> Tuple[float, str]:
        if not candidate.education:
            return 0.0, "No education information found on the resume."

        candidate_rank = max(
            (self._degree_ranks.get(e.degree_level, -1) for e in candidate.education), default=-1
        )
        top_level = next(
            (e.degree_level for e in candidate.education if self._degree_ranks.get(e.degree_level, -1) == candidate_rank),
            "an unspecified degree",
        )

        if not jd.required_education_level:
            return 80.0, f"No specific degree level required by the JD; candidate holds {top_level or 'a degree'}."

        required_rank = self._degree_ranks.get(jd.required_education_level, -1)
        if candidate_rank >= required_rank:
            return 100.0, f"Candidate's {top_level} meets or exceeds the required {jd.required_education_level}."

        gap = required_rank - candidate_rank
        score = clamp(100 - gap * 30)
        return score, f"Candidate's {top_level} is below the required {jd.required_education_level}."

    def _score_certifications(self, candidate: Candidate, jd: JobDescription) -> Tuple[float, str]:
        if not jd.required_certifications:
            if candidate.certifications:
                avg_weight = self._average_cert_weight(candidate.certifications)
                score = clamp(50 + avg_weight * 40)
                return score, (
                    f"No certifications specifically required; candidate holds "
                    f"{len(candidate.certifications)} certification(s)."
                )
            return 50.0, "No certifications required by the JD, and none held."

        matched = [
            req for req in jd.required_certifications
            if any(normalize_for_matching(req) in normalize_for_matching(held) for held in candidate.certifications)
        ]
        score = clamp(safe_divide(len(matched), len(jd.required_certifications)) * 100)
        return score, f"Matched {len(matched)}/{len(jd.required_certifications)} required certification(s)."

    def _average_cert_weight(self, certifications: List[str]) -> float:
        if not certifications:
            return 0.0
        weights = []
        for cert in certifications:
            for provider, info in self._cert_provider_weights.items():
                if provider.lower() in cert.lower():
                    weights.append(info)
                    break
            else:
                weights.append(0.5)  # unrecognized provider — moderate default
        return sum(weights) / len(weights)

    @staticmethod
    def _score_projects(candidate: Candidate, jd: JobDescription) -> Tuple[float, str]:
        if not candidate.projects:
            return 0.0, "No projects listed on the resume."

        relevant_skills = set(jd.required_skills) | set(jd.preferred_skills)
        per_project_scores = []
        for project in candidate.projects:
            score = 0.0
            # Deliberately withholds credit from title-only projects (per
            # the spec: "avoid giving full credit to projects with only titles").
            if project.description and len(project.description.split()) >= _MIN_PROJECT_DESCRIPTION_WORDS:
                score += 40
            if project.has_deployment_evidence:
                score += 30
            if relevant_skills:
                if set(project.tech_stack) & relevant_skills:
                    score += 30
            else:
                score += 15
            per_project_scores.append(min(score, 100))

        average = sum(per_project_scores) / len(per_project_scores)
        count_bonus = min(len(candidate.projects), _MAX_PROJECT_COUNT_BONUS_AT) / _MAX_PROJECT_COUNT_BONUS_AT * 10
        final = clamp(average * 0.9 + count_bonus)
        reason = (
            f"{len(candidate.projects)} project(s) evaluated; average quality {average:.0f}/100 "
            f"based on description depth, deployment evidence, and relevance to the JD's tech stack."
        )
        return final, reason

    # ------------------------------------------------------------------ #
    # Recommendation / confidence banding
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_recommendation(overall_score: float) -> str:
        for threshold, label in RECOMMENDATION_BANDS:
            if overall_score >= threshold:
                return label
        return RECOMMENDATION_BANDS[-1][1]

    @staticmethod
    def _categorize_confidence(confidence: Optional[float]) -> str:
        if confidence is None:
            return "Low"
        if confidence >= CONFIDENCE_THRESHOLD_HIGH:
            return "High"
        if confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
            return "Medium"
        return "Low"

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_degree_ranks() -> Dict[str, int]:
        data = json.loads(DEGREES_FILE.read_text(encoding="utf-8"))
        return {entry["level"]: entry["rank"] for entry in data["degree_levels"]}

    @staticmethod
    def _load_cert_provider_weights() -> Dict[str, float]:
        data = json.loads(CERTIFICATIONS_FILE.read_text(encoding="utf-8"))
        return {provider: info["weight"] for provider, info in data["providers"].items()}


# --------------------------------------------------------------------------- #
# Recruiter-facing narrative
# --------------------------------------------------------------------------- #
class InsightsGenerator:
    """Generates strengths/weaknesses/improvement-areas/summary from a
    candidate's already-computed RankingResult + MatchingResult. Purely
    template-based and deterministic (no LLM call here) so every sentence
    is directly traceable to a computed score  nothing is invented. An
    LLM-powered version of this narrative is the AI Chatbot's job (Module 5),
    which will be instructed to draw only from this same structured data.
    """

    _STRONG_THRESHOLD = 75
    _WEAK_THRESHOLD = 50
    _CATEGORY_LABELS = {
        "skills": "skill match", "experience": "experience", "education": "education",
        "certification": "certifications", "project": "project portfolio",
    }

    def generate(self, candidate: Candidate) -> AIInsights:
        return AIInsights(
            strengths=self._build_strengths(candidate),
            weaknesses=self._build_weaknesses(candidate),
            improvement_areas=self._build_improvement_areas(candidate),
            skill_match_explanation=candidate.matching.explanation,
            recruiter_summary=self._build_summary(candidate),
        )

    def _build_strengths(self, candidate: Candidate) -> List[str]:
        strengths: List[str] = []
        breakdown = candidate.ranking.category_breakdown
        matching = candidate.matching

        skills_cat = breakdown.get("skills")
        if skills_cat and skills_cat.raw_score >= self._STRONG_THRESHOLD and matching.matched_skills:
            top = ", ".join(matching.matched_skills[:5])
            strengths.append(f"Strong skill alignment ({skills_cat.raw_score:.0f}/100) — demonstrates {top}.")

        experience_cat = breakdown.get("experience")
        if experience_cat and experience_cat.raw_score >= self._STRONG_THRESHOLD:
            seniority_note = f", seniority level: {candidate.seniority_level}" if candidate.seniority_level else ""
            strengths.append(
                f"Meets or exceeds experience requirements ({candidate.total_years_experience:.1f} years{seniority_note})."
            )

        education_cat = breakdown.get("education")
        if education_cat and education_cat.raw_score >= self._STRONG_THRESHOLD:
            strengths.append(education_cat.reason)

        cert_cat = breakdown.get("certification")
        if cert_cat and cert_cat.raw_score >= self._STRONG_THRESHOLD and candidate.certifications:
            strengths.append(f"Holds relevant certification(s): {', '.join(candidate.certifications[:3])}.")

        project_cat = breakdown.get("project")
        if project_cat and project_cat.raw_score >= self._STRONG_THRESHOLD:
            strengths.append(
                f"Strong project portfolio ({len(candidate.projects)} project(s) with clear relevance and evidence)."
            )

        if matching.extra_skills:
            strengths.append(
                f"Brings {len(matching.extra_skills)} additional skill(s) beyond the JD's stated requirements."
            )

        return strengths or ["No standout strength categories identified relative to this job description."]

    def _build_weaknesses(self, candidate: Candidate) -> List[str]:
        weaknesses: List[str] = []
        breakdown = candidate.ranking.category_breakdown
        matching = candidate.matching

        if matching.missing_required:
            weaknesses.append(f"Missing required skill(s): {', '.join(matching.missing_required[:5])}.")

        for category in ("experience", "education", "certification", "project"):
            cat_score = breakdown.get(category)
            if cat_score and cat_score.raw_score < self._WEAK_THRESHOLD:
                weaknesses.append(cat_score.reason)

        if candidate.diagnostics.warnings:
            weaknesses.append(
                f"Resume parsing raised {len(candidate.diagnostics.warnings)} warning(s) — "
                "some data may be incomplete."
            )

        return weaknesses or ["No significant weaknesses identified relative to this job description."]

    def _build_improvement_areas(self, candidate: Candidate) -> List[str]:
        areas: List[str] = []
        matching = candidate.matching
        ranking = candidate.ranking

        if matching.missing_required:
            areas.append(
                f"Gaining hands-on experience with {', '.join(matching.missing_required[:3])} "
                "would close the largest skill gap(s)."
            )
        if matching.missing_preferred:
            areas.append(
                f"Additional exposure to {', '.join(matching.missing_preferred[:3])} "
                "would strengthen alignment with preferred qualifications."
            )
        if ranking.experience_score < self._WEAK_THRESHOLD:
            areas.append("Additional relevant work experience would improve alignment with this role's expectations.")
        if ranking.certification_score < self._WEAK_THRESHOLD and not candidate.certifications:
            areas.append("A recognized certification relevant to this role could strengthen the application.")

        return areas or ["No specific improvement areas identified — candidate appears well-aligned with this role."]

    def _build_summary(self, candidate: Candidate) -> str:
        ranking = candidate.ranking
        breakdown = ranking.category_breakdown
        if not breakdown:
            return f"Overall score {ranking.overall_score:.0f}/100 ({ranking.recommendation})."

        strong = sorted(
            (c for c, s in breakdown.items() if s.raw_score >= self._STRONG_THRESHOLD),
            key=lambda c: breakdown[c].raw_score, reverse=True,
        )
        weak = sorted(
            (c for c, s in breakdown.items() if s.raw_score < self._WEAK_THRESHOLD),
            key=lambda c: breakdown[c].raw_score,
        )

        parts = [f"Overall score {ranking.overall_score:.0f}/100 ({ranking.recommendation})."]
        if strong:
            labels = ", ".join(self._CATEGORY_LABELS[c] for c in strong[:3])
            parts.append(f"Strong on {labels}.")
        if weak:
            labels = ", ".join(self._CATEGORY_LABELS[c] for c in weak[:2])
            parts.append(f"Gaps in {labels}.")
        if not strong and not weak:
            top_category, top_score = max(breakdown.items(), key=lambda kv: kv[1].raw_score)
            parts.append(f"Strongest in {self._CATEGORY_LABELS[top_category]} ({top_score.raw_score:.0f}/100).")
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Batch orchestration
# --------------------------------------------------------------------------- #
def rank_candidates(
    candidates: List[Candidate],
    job_description: JobDescription,
    engine: Optional[RankingEngine] = None,
    insights_generator: Optional[InsightsGenerator] = None,
) -> List[Candidate]:
    """Score every candidate, assign 1-based ranks by descending overall
    score, generate recruiter insights, and return the list SORTED by rank.

    Assumes candidate.matching has already been populated  see
    core.matcher.match_candidates(), which must run first.
    """
    engine = engine or RankingEngine()
    insights_generator = insights_generator or InsightsGenerator()

    for candidate in candidates:
        candidate.diagnostics.status = ProcessingStatus.RANKED
        if candidate.ranking.rank is not None:  # a real prior ranking, not the untouched default
            candidate.ranking_history.append(RankingSnapshot(
                overall_score=candidate.ranking.overall_score,
                recommendation=candidate.ranking.recommendation,
                scoring_version=candidate.ranking.scoring_version,
                weights_snapshot=dict(candidate.ranking.weights_snapshot),
            ))
        candidate.ranking = engine.rank(candidate, job_description)

    ranked = sorted(candidates, key=lambda c: c.ranking.overall_score, reverse=True)
    for position, candidate in enumerate(ranked, start=1):
        candidate.ranking.rank = position
        candidate.insights = insights_generator.generate(candidate)
        candidate.diagnostics.status = ProcessingStatus.COMPLETED

    logger.info("Ranked %d candidates for JD '%s'", len(ranked), job_description.title)
    return ranked


# --------------------------------------------------------------------------- #
# Batch statistics
# --------------------------------------------------------------------------- #
@dataclass
class BatchRankingStatistics:
    """Aggregate view of overall scores across a ranked batch — the numbers
    a dashboard's summary row needs, computed once here rather than ad hoc
    in the UI layer. Pairs with core.matcher.BatchMatchStatistics, which
    covers the pre-ranking match signals.
    """
    candidate_count: int = 0
    average_score: float = 0.0
    highest_score: float = 0.0
    lowest_score: float = 0.0
    median_score: float = 0.0
    std_deviation: float = 0.0
    recommendation_counts: Dict[str, int] = field(default_factory=dict)


def compute_batch_statistics(candidates: List[Candidate]) -> BatchRankingStatistics:
    """Summarize ranking results across a batch. Safe to call on an empty
    list or on candidates that haven't been ranked yet (all-zero stats).
    """
    if not candidates:
        return BatchRankingStatistics()

    scores = [c.ranking.overall_score for c in candidates]
    recommendation_counts: Dict[str, int] = {}
    for candidate in candidates:
        label = candidate.ranking.recommendation or "Unranked"
        recommendation_counts[label] = recommendation_counts.get(label, 0) + 1

    return BatchRankingStatistics(
        candidate_count=len(candidates),
        average_score=round(statistics.mean(scores), 1),
        highest_score=round(max(scores), 1),
        lowest_score=round(min(scores), 1),
        median_score=round(statistics.median(scores), 1),
        std_deviation=round(statistics.stdev(scores), 1) if len(scores) > 1 else 0.0,
        recommendation_counts=recommendation_counts,
    )
