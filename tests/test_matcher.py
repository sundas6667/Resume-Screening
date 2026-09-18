"""Unit tests for core/matcher.py."""
import pytest

from core.matcher import SkillMatcher, match_candidates
from core.vector_store import VectorStore
from models.candidate import Candidate
from models.job_description import JobDescription


@pytest.fixture
def matcher() -> SkillMatcher:
    return SkillMatcher()


def make_candidate(skills, confidence=90.0, raw_text="") -> Candidate:
    c = Candidate(full_name="Test Candidate", skills=skills, raw_text=raw_text)
    c.diagnostics.confidence = confidence
    return c


def make_jd(required=None, preferred=None, cleaned_text="") -> JobDescription:
    return JobDescription(
        raw_text=cleaned_text, cleaned_text=cleaned_text,
        required_skills=required or [], preferred_skills=preferred or [],
    )


class TestSkillMatching:
    def test_all_required_skills_matched(self, matcher):
        candidate = make_candidate(["Python", "AWS", "Docker"])
        jd = make_jd(required=["Python", "AWS", "Docker"])
        result = matcher.match(candidate, jd)
        assert result.matched_skills == ["AWS", "Docker", "Python"]
        assert result.missing_skills == []

    def test_missing_skills_identified(self, matcher):
        candidate = make_candidate(["Python"])
        jd = make_jd(required=["Python", "AWS", "Kubernetes"])
        result = matcher.match(candidate, jd)
        assert result.matched_skills == ["Python"]
        assert set(result.missing_skills) == {"AWS", "Kubernetes"}

    def test_extra_skills_identified(self, matcher):
        candidate = make_candidate(["Python", "React", "Vue.js"])
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        assert set(result.extra_skills) == {"React", "Vue.js"}

    def test_required_and_preferred_both_considered_for_matching(self, matcher):
        candidate = make_candidate(["Python", "FastAPI"])
        jd = make_jd(required=["Python"], preferred=["FastAPI"])
        result = matcher.match(candidate, jd)
        assert set(result.matched_skills) == {"Python", "FastAPI"}

    def test_no_overlap_yields_all_missing(self, matcher):
        candidate = make_candidate(["React", "Node.js"])
        jd = make_jd(required=["Python", "AWS"])
        result = matcher.match(candidate, jd)
        assert result.matched_skills == []
        assert set(result.missing_skills) == {"Python", "AWS"}

    def test_empty_candidate_skills(self, matcher):
        candidate = make_candidate([])
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        assert result.matched_skills == []
        assert result.missing_skills == ["Python"]


class TestKeywordMatching:
    def test_matched_keyword_found_in_resume_text(self, matcher):
        candidate = make_candidate(["Python"], raw_text="Experienced with regulatory compliance and audits.")
        jd = make_jd(required=["Python"], cleaned_text="Need Python and regulatory compliance experience.")
        result = matcher.match(candidate, jd)
        assert "regulatory" in result.matched_keywords or "compliance" in result.matched_keywords

    def test_skill_words_excluded_from_keywords(self, matcher):
        candidate = make_candidate(["Python"], raw_text="I know Python well.")
        jd = make_jd(required=["Python"], cleaned_text="Need Python experience.")
        result = matcher.match(candidate, jd)
        assert "python" not in [k.lower() for k in result.matched_keywords]
        assert "python" not in [k.lower() for k in result.missing_keywords]

    def test_missing_keyword_not_in_resume(self, matcher):
        candidate = make_candidate(["Python"], raw_text="I know Python.")
        jd = make_jd(required=["Python"], cleaned_text="Need Python and leadership mentorship abilities.")
        result = matcher.match(candidate, jd)
        assert "leadership" in result.missing_keywords or "mentorship" in result.missing_keywords


class TestExplanation:
    def test_explanation_mentions_matched_and_missing(self, matcher):
        candidate = make_candidate(["Python", "AWS"])
        jd = make_jd(required=["Python", "AWS", "Kubernetes"])
        result = matcher.match(candidate, jd)
        assert "Python" in result.explanation or "AWS" in result.explanation
        assert "Kubernetes" in result.explanation
        assert "%" in result.explanation

    def test_explanation_handles_no_requirements(self, matcher):
        candidate = make_candidate(["Python"])
        jd = make_jd()
        result = matcher.match(candidate, jd)
        assert "No specific skill requirements" in result.explanation

    def test_explanation_mentions_extra_skills(self, matcher):
        candidate = make_candidate(["Python", "React", "Vue.js"])
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        assert "additional skill" in result.explanation


class TestMatchConfidence:
    def test_confidence_reflects_candidate_parsing_confidence(self, matcher):
        candidate = make_candidate(["Python"], confidence=42.0)
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        assert result.confidence == 42.0

    def test_low_confidence_when_jd_has_no_skills(self, matcher):
        candidate = make_candidate(["Python"], confidence=95.0)
        jd = make_jd()
        result = matcher.match(candidate, jd)
        assert result.confidence == 30.0


class TestSimilarityScorePassthrough:
    def test_similarity_score_stored_on_result(self, matcher):
        candidate = make_candidate(["Python"])
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd, similarity_score=77.3)
        assert result.similarity_score == 77.3

    def test_default_similarity_is_zero(self, matcher):
        candidate = make_candidate(["Python"])
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        assert result.similarity_score == 0.0


class TestBatchOrchestration:
    def test_populates_matching_for_every_candidate(self):
        candidates = [
            make_candidate(["Python", "AWS"], raw_text="Python AWS expert with cloud experience"),
            make_candidate(["React"], raw_text="React frontend developer"),
        ]
        jd = make_jd(required=["Python", "AWS"], cleaned_text="Need Python and AWS cloud experience")
        result = match_candidates(candidates, jd, vector_store=VectorStore())
        assert all(c.matching.matched_skills or c.matching.missing_skills for c in result)

    def test_similarity_scores_populated_when_vector_store_provided(self):
        candidates = [
            make_candidate(["Python", "AWS"], raw_text="Python AWS cloud infrastructure expert engineer"),
            make_candidate(["React"], raw_text="React JavaScript frontend web developer"),
        ]
        jd = make_jd(required=["Python", "AWS"], cleaned_text="Need Python AWS cloud infrastructure engineer")
        match_candidates(candidates, jd, vector_store=VectorStore())
        assert candidates[0].matching.similarity_score > candidates[1].matching.similarity_score

    def test_works_without_vector_store(self):
        candidates = [make_candidate(["Python"])]
        jd = make_jd(required=["Python"])
        result = match_candidates(candidates, jd, vector_store=None)
        assert result[0].matching.similarity_score == 0.0
        assert result[0].matching.matched_skills == ["Python"]

    def test_handles_candidate_with_empty_raw_text_gracefully(self):
        candidates = [make_candidate(["Python"], raw_text="")]
        jd = make_jd(required=["Python"], cleaned_text="Need Python")
        result = match_candidates(candidates, jd, vector_store=VectorStore())
        assert result[0].matching.matched_skills == ["Python"]

    def test_empty_candidate_list_returns_empty(self):
        jd = make_jd(required=["Python"])
        assert match_candidates([], jd, vector_store=VectorStore()) == []


class TestRequiredVsPreferredSeparation:
    def test_matched_required_and_preferred_kept_separate(self, matcher):
        candidate = make_candidate(["Python", "Docker"])
        jd = make_jd(required=["Python", "Kubernetes"], preferred=["Docker", "Terraform"])
        result = matcher.match(candidate, jd)
        assert result.matched_required == ["Python"]
        assert result.matched_preferred == ["Docker"]
        assert result.missing_required == ["Kubernetes"]
        assert result.missing_preferred == ["Terraform"]

    def test_combined_fields_still_populated_for_simple_display(self, matcher):
        candidate = make_candidate(["Python", "Docker"])
        jd = make_jd(required=["Python", "Kubernetes"], preferred=["Docker", "Terraform"])
        result = matcher.match(candidate, jd)
        assert set(result.matched_skills) == {"Python", "Docker"}
        assert set(result.missing_skills) == {"Kubernetes", "Terraform"}


class TestConfidenceReason:
    def test_reason_mentions_parsing_confidence(self, matcher):
        candidate = make_candidate(["Python"], confidence=91.0)
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        assert "91" in result.confidence_reason

    def test_reason_flags_parsing_warnings(self, matcher):
        candidate = make_candidate(["Python"], confidence=60.0)
        candidate.diagnostics.add_warning("Education section not found")
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        assert "warning" in result.confidence_reason.lower()

    def test_reason_explains_missing_jd_skills_case(self, matcher):
        candidate = make_candidate(["Python"])
        jd = make_jd()
        result = matcher.match(candidate, jd)
        assert "no skill requirements" in result.confidence_reason.lower()


class TestZeroOverlapHandling:
    """Regression test for the reviewer's exact scenario: a candidate whose
    skills share nothing at all with the JD must degrade gracefully, never crash."""

    def test_zero_overlap_does_not_crash_and_produces_valid_result(self, matcher):
        candidate = make_candidate(
            ["Leadership", "Project Management"],
            raw_text="Experienced Manager with an MBA. Strong Leadership and Project Management skills.",
        )
        jd = make_jd(required=["Python", "TensorFlow"], cleaned_text="Need Python and TensorFlow experience.")
        result = matcher.match(candidate, jd)
        assert result.matched_skills == []
        assert set(result.missing_skills) == {"Python", "TensorFlow"}
        assert "Overall skill match: 0%" in result.explanation

    def test_zero_overlap_explanation_is_not_awkwardly_redundant(self, matcher):
        candidate = make_candidate(["Leadership"])
        jd = make_jd(required=["Python"])
        result = matcher.match(candidate, jd)
        # Should not contain the old "shows none... but is missing" double-negative phrasing
        assert "shows none" in result.explanation.lower()
        assert result.explanation.count("missing") <= 1

    def test_zero_overlap_full_pipeline_via_match_candidates(self):
        candidate = make_candidate(
            ["Leadership", "Project Management"],
            raw_text="Experienced Manager with an MBA. Strong Leadership and Project Management skills.",
        )
        jd = make_jd(required=["Python", "TensorFlow"], cleaned_text="Need Python and TensorFlow experience.")
        result = match_candidates([candidate], jd, vector_store=VectorStore())
        assert result[0].matching.similarity_score == 0.0
        assert result[0].matching.matched_skills == []


class TestCategorizeSkills:
    def test_groups_skills_by_taxonomy_category(self):
        from core.extractor import get_skill_database
        from core.matcher import categorize_skills

        grouped = categorize_skills(["Python", "Docker", "AWS"], get_skill_database())
        assert "Programming Languages" in grouped
        assert "Python" in grouped["Programming Languages"]
        assert "Cloud & DevOps" in grouped
        assert set(grouped["Cloud & DevOps"]) == {"Docker", "AWS"}

    def test_empty_skill_list_returns_empty_dict(self):
        from core.extractor import get_skill_database
        from core.matcher import categorize_skills

        assert categorize_skills([], get_skill_database()) == {}


class TestCategorizeSimilarity:
    def test_high_medium_low_bands(self):
        from core.matcher import categorize_similarity

        assert categorize_similarity(80.0) == "High"
        assert categorize_similarity(20.0) == "Medium"
        assert categorize_similarity(2.0) == "Low"


class TestBatchStatistics:
    def test_empty_candidate_list_returns_zeroed_stats(self):
        from core.matcher import compute_batch_statistics

        stats = compute_batch_statistics([], make_jd(required=["Python"]))
        assert stats.candidate_count == 0
        assert stats.average_similarity == 0.0

    def test_computes_correct_aggregates(self):
        from core.matcher import compute_batch_statistics

        jd = make_jd(required=["Python", "AWS"], cleaned_text="Need Python and AWS")
        candidates = [
            make_candidate(["Python", "AWS"], raw_text="Python AWS expert"),
            make_candidate(["Python"], raw_text="Python developer"),
        ]
        match_candidates(candidates, jd, vector_store=VectorStore())
        stats = compute_batch_statistics(candidates, jd)
        assert stats.candidate_count == 2
        assert stats.highest_similarity >= stats.lowest_similarity
        assert 0.0 <= stats.average_skill_coverage <= 100.0

    def test_vocabulary_size_included_when_vector_store_provided(self):
        from core.matcher import compute_batch_statistics

        jd = make_jd(required=["Python"], cleaned_text="Need Python experience")
        candidates = [make_candidate(["Python"], raw_text="Python developer with years of experience")]
        vs = VectorStore()
        match_candidates(candidates, jd, vector_store=vs)
        stats = compute_batch_statistics(candidates, jd, vector_store=vs)
        assert stats.vocabulary_size is not None and stats.vocabulary_size > 0
