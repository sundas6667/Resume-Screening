"""Unit tests for core/ranking.py."""
import pytest

from config import ScoringWeights
from core.ranking import InsightsGenerator, RankingEngine, rank_candidates
from models.candidate import Candidate, Education, ExperienceEntry, MatchingResult, ProjectEntry
from models.job_description import JobDescription


@pytest.fixture
def engine() -> RankingEngine:
    return RankingEngine()


def make_candidate(
    full_name="Test Candidate", skills=None, education=None, experience=None, projects=None,
    certifications=None, total_years=0.0, matching=None, confidence=90.0,
) -> Candidate:
    c = Candidate(
        full_name=full_name,
        skills=skills or [],
        education=education or [],
        experience=experience or [],
        projects=projects or [],
        certifications=certifications or [],
        total_years_experience=total_years,
    )
    c.diagnostics.confidence = confidence
    c.matching = matching or MatchingResult()
    return c


def make_jd(required_skills=None, preferred_skills=None, min_years=None, education_level=None, certs=None) -> JobDescription:
    return JobDescription(
        raw_text="jd", cleaned_text="jd",
        required_skills=required_skills or [], preferred_skills=preferred_skills or [],
        min_years_experience=min_years, required_education_level=education_level,
        required_certifications=certs or [],
    )


class TestSkillScoring:
    def test_all_required_skills_matched_scores_high(self, engine):
        matching = MatchingResult(matched_required=["Python", "AWS"], missing_required=[])
        candidate = make_candidate(matching=matching)
        jd = make_jd(required_skills=["Python", "AWS"])
        score, reason = engine._score_skills(candidate, jd)
        assert score >= 65  # keyword component alone is 100%, blended with 0 similarity -> 70

    def test_no_required_skills_matched_scores_low(self, engine):
        matching = MatchingResult(matched_required=[], missing_required=["Python", "AWS"])
        candidate = make_candidate(matching=matching)
        jd = make_jd(required_skills=["Python", "AWS"])
        score, _ = engine._score_skills(candidate, jd)
        assert score < 20

    def test_semantic_similarity_contributes_to_score(self, engine):
        matching_low_sim = MatchingResult(matched_required=["Python"], similarity_score=0.0)
        matching_high_sim = MatchingResult(matched_required=["Python"], similarity_score=100.0)
        jd = make_jd(required_skills=["Python"])
        score_low, _ = engine._score_skills(make_candidate(matching=matching_low_sim), jd)
        score_high, _ = engine._score_skills(make_candidate(matching=matching_high_sim), jd)
        assert score_high > score_low

    def test_no_skills_in_jd_at_all(self, engine):
        candidate = make_candidate(matching=MatchingResult())
        jd = make_jd()
        score, reason = engine._score_skills(candidate, jd)
        assert score == 0.0
        assert "no skill requirements" in reason.lower()

    def test_preferred_skill_bonus_applied(self, engine):
        base_matching = MatchingResult(matched_required=["Python"], matched_preferred=[])
        bonus_matching = MatchingResult(matched_required=["Python"], matched_preferred=["Docker"])
        jd = make_jd(required_skills=["Python"], preferred_skills=["Docker"])
        base_score, _ = engine._score_skills(make_candidate(matching=base_matching), jd)
        bonus_score, _ = engine._score_skills(make_candidate(matching=bonus_matching), jd)
        assert bonus_score > base_score


class TestExperienceScoring:
    def test_meets_minimum_years_scores_100(self, engine):
        candidate = make_candidate(total_years=6.0)
        jd = make_jd(min_years=5.0)
        score, _ = engine._score_experience(candidate, jd)
        assert score == 100.0

    def test_below_minimum_years_scores_proportionally(self, engine):
        candidate = make_candidate(total_years=2.5)
        jd = make_jd(min_years=5.0)
        score, _ = engine._score_experience(candidate, jd)
        assert score == 50.0

    def test_zero_years_scores_zero(self, engine):
        candidate = make_candidate(total_years=0.0)
        jd = make_jd(min_years=5.0)
        score, _ = engine._score_experience(candidate, jd)
        assert score == 0.0

    def test_no_minimum_specified_falls_back_to_reference_scale(self, engine):
        candidate = make_candidate(total_years=10.0)
        jd = make_jd()
        score, reason = engine._score_experience(candidate, jd)
        assert score == 100.0
        assert "did not specify" in reason.lower()

    def test_score_never_exceeds_100_for_overqualified_candidate(self, engine):
        candidate = make_candidate(total_years=30.0)
        jd = make_jd(min_years=5.0)
        score, _ = engine._score_experience(candidate, jd)
        assert score == 100.0


class TestEducationScoring:
    def test_no_education_scores_zero(self, engine):
        candidate = make_candidate(education=[])
        jd = make_jd(education_level="Bachelor")
        score, reason = engine._score_education(candidate, jd)
        assert score == 0.0

    def test_meets_required_level_scores_100(self, engine):
        candidate = make_candidate(education=[Education(degree_level="Master")])
        jd = make_jd(education_level="Bachelor")
        score, _ = engine._score_education(candidate, jd)
        assert score == 100.0

    def test_exact_match_scores_100(self, engine):
        candidate = make_candidate(education=[Education(degree_level="Bachelor")])
        jd = make_jd(education_level="Bachelor")
        score, _ = engine._score_education(candidate, jd)
        assert score == 100.0

    def test_below_required_level_scores_partial(self, engine):
        candidate = make_candidate(education=[Education(degree_level="Bachelor")])
        jd = make_jd(education_level="PhD")
        score, _ = engine._score_education(candidate, jd)
        assert 0 < score < 100

    def test_no_requirement_specified_gives_baseline_credit(self, engine):
        candidate = make_candidate(education=[Education(degree_level="Bachelor")])
        jd = make_jd()
        score, _ = engine._score_education(candidate, jd)
        assert score == 80.0

    def test_highest_degree_used_when_multiple(self, engine):
        candidate = make_candidate(education=[
            Education(degree_level="Bachelor"), Education(degree_level="Master"),
        ])
        jd = make_jd(education_level="Master")
        score, _ = engine._score_education(candidate, jd)
        assert score == 100.0


class TestCertificationScoring:
    def test_no_requirement_and_no_certs(self, engine):
        candidate = make_candidate(certifications=[])
        jd = make_jd()
        score, _ = engine._score_certifications(candidate, jd)
        assert score == 50.0

    def test_no_requirement_but_holds_certs_gets_bonus(self, engine):
        candidate = make_candidate(certifications=["AWS Certified Solutions Architect"])
        jd = make_jd()
        score, _ = engine._score_certifications(candidate, jd)
        assert score > 50.0

    def test_matches_required_certification(self, engine):
        candidate = make_candidate(certifications=["AWS Certified Solutions Architect"])
        jd = make_jd(certs=["AWS"])
        score, _ = engine._score_certifications(candidate, jd)
        assert score == 100.0

    def test_missing_required_certification(self, engine):
        candidate = make_candidate(certifications=[])
        jd = make_jd(certs=["AWS", "Azure"])
        score, _ = engine._score_certifications(candidate, jd)
        assert score == 0.0

    def test_partial_certification_match(self, engine):
        candidate = make_candidate(certifications=["AWS Certified Solutions Architect"])
        jd = make_jd(certs=["AWS", "Azure"])
        score, _ = engine._score_certifications(candidate, jd)
        assert score == 50.0


class TestProjectScoring:
    def test_no_projects_scores_zero(self, engine):
        candidate = make_candidate(projects=[])
        jd = make_jd()
        score, _ = engine._score_projects(candidate, jd)
        assert score == 0.0

    def test_title_only_project_scores_low(self, engine):
        """Regression guard for the spec's explicit instruction: don't give
        full credit to projects with only a title."""
        candidate = make_candidate(projects=[ProjectEntry(name="My Project", description="")])
        jd = make_jd(required_skills=["Python"])
        score, _ = engine._score_projects(candidate, jd)
        assert score < 30

    def test_well_documented_relevant_project_scores_high(self, engine):
        candidate = make_candidate(projects=[ProjectEntry(
            name="ML Pipeline",
            description="Built an end-to-end machine learning pipeline for real-time fraud detection at scale.",
            tech_stack=["Python", "TensorFlow"],
            github_link="github.com/x/y",
        )])
        jd = make_jd(required_skills=["Python", "TensorFlow"])
        score, _ = engine._score_projects(candidate, jd)
        assert score >= 90

    def test_multiple_projects_get_count_bonus(self, engine):
        one_project = [ProjectEntry(name="A", description="A solid, well-described project with real depth here.",
                                     tech_stack=["Python"], github_link="x")]
        three_projects = one_project * 3
        jd = make_jd(required_skills=["Python"])
        score_one, _ = engine._score_projects(make_candidate(projects=one_project), jd)
        score_three, _ = engine._score_projects(make_candidate(projects=three_projects), jd)
        assert score_three > score_one


class TestRecommendationBanding:
    def test_high_score_highly_recommended(self, engine):
        assert engine._get_recommendation(90) == "Highly Recommended"

    def test_mid_high_score_recommended(self, engine):
        assert engine._get_recommendation(65) == "Recommended"

    def test_mid_low_score_consider(self, engine):
        assert engine._get_recommendation(45) == "Consider"

    def test_low_score_not_recommended(self, engine):
        assert engine._get_recommendation(10) == "Not Recommended"

    def test_boundary_values(self, engine):
        assert engine._get_recommendation(80) == "Highly Recommended"
        assert engine._get_recommendation(60) == "Recommended"
        assert engine._get_recommendation(40) == "Consider"
        assert engine._get_recommendation(39.9) == "Not Recommended"


class TestConfidenceCategorization:
    def test_high_confidence(self, engine):
        assert engine._categorize_confidence(90) == "High"

    def test_medium_confidence(self, engine):
        assert engine._categorize_confidence(60) == "Medium"

    def test_low_confidence(self, engine):
        assert engine._categorize_confidence(20) == "Low"

    def test_none_confidence_defaults_low(self, engine):
        assert engine._categorize_confidence(None) == "Low"


class TestFullRankMethod:
    def test_overall_score_within_0_to_100(self, engine):
        candidate = make_candidate(
            skills=["Python"], education=[Education(degree_level="Bachelor")],
            experience=[ExperienceEntry(start_year=2020, is_current=True)],
            projects=[ProjectEntry(name="X", description="A reasonably detailed project description here.")],
            certifications=["AWS Certified"], total_years=4.0,
            matching=MatchingResult(matched_required=["Python"], similarity_score=50.0),
        )
        jd = make_jd(required_skills=["Python"], min_years=3.0)
        result = engine.rank(candidate, jd)
        assert 0.0 <= result.overall_score <= 100.0

    def test_weights_sum_reflected_in_category_breakdown(self, engine):
        candidate = make_candidate(matching=MatchingResult(matched_required=["Python"]))
        jd = make_jd(required_skills=["Python"])
        result = engine.rank(candidate, jd)
        total_weight = sum(cs.weight for cs in result.category_breakdown.values())
        assert total_weight == pytest.approx(1.0)

    def test_overall_score_equals_sum_of_weighted_contributions(self, engine):
        candidate = make_candidate(
            skills=["Python"], matching=MatchingResult(matched_required=["Python"], similarity_score=50.0),
        )
        jd = make_jd(required_skills=["Python"])
        result = engine.rank(candidate, jd)
        summed = sum(cs.weighted_contribution for cs in result.category_breakdown.values())
        assert result.overall_score == pytest.approx(summed, abs=0.2)

    def test_custom_weights_respected(self):
        heavy_skills_weights = ScoringWeights(skills=0.9, experience=0.025, education=0.025, certification=0.025, project=0.025)
        engine = RankingEngine(weights=heavy_skills_weights)
        candidate = make_candidate(matching=MatchingResult(matched_required=["Python"], similarity_score=100.0))
        jd = make_jd(required_skills=["Python"])
        result = engine.rank(candidate, jd)
        # With 90% weight on a maxed-out skill score, overall should be high
        assert result.overall_score > 70


class TestInsightsGenerator:
    @pytest.fixture
    def generator(self) -> InsightsGenerator:
        return InsightsGenerator()

    def _ranked_candidate(self, engine, **kwargs) -> Candidate:
        candidate = make_candidate(**kwargs)
        jd = make_jd(required_skills=["Python", "AWS"], min_years=3.0)
        candidate.ranking = engine.rank(candidate, jd)
        return candidate

    def test_strengths_generated_for_strong_candidate(self, engine, generator):
        candidate = self._ranked_candidate(
            engine, skills=["Python", "AWS"], total_years=6.0,
            matching=MatchingResult(matched_required=["Python", "AWS"], similarity_score=80.0),
        )
        insights = generator.generate(candidate)
        assert len(insights.strengths) > 0
        assert insights.strengths[0] != "No standout strength categories identified relative to this job description."

    def test_weaknesses_generated_for_weak_candidate(self, engine, generator):
        candidate = self._ranked_candidate(
            engine, skills=[], total_years=0.0,
            matching=MatchingResult(missing_required=["Python", "AWS"]),
        )
        insights = generator.generate(candidate)
        assert any("missing" in w.lower() for w in insights.weaknesses)

    def test_recruiter_summary_mentions_overall_score(self, engine, generator):
        candidate = self._ranked_candidate(
            engine, skills=["Python"], matching=MatchingResult(matched_required=["Python"]),
        )
        insights = generator.generate(candidate)
        assert str(int(candidate.ranking.overall_score)) in insights.recruiter_summary or "score" in insights.recruiter_summary.lower()

    def test_improvement_areas_reference_missing_skills(self, engine, generator):
        candidate = self._ranked_candidate(
            engine, skills=["Python"],
            matching=MatchingResult(matched_required=["Python"], missing_required=["AWS"]),
        )
        insights = generator.generate(candidate)
        assert any("AWS" in area for area in insights.improvement_areas)


class TestRankCandidatesOrchestration:
    def test_assigns_sequential_ranks_by_descending_score(self):
        strong = make_candidate(
            full_name="Strong", skills=["Python", "AWS"], total_years=6.0,
            matching=MatchingResult(matched_required=["Python", "AWS"], similarity_score=80.0),
        )
        weak = make_candidate(
            full_name="Weak", skills=[], total_years=0.0, matching=MatchingResult(missing_required=["Python", "AWS"]),
        )
        jd = make_jd(required_skills=["Python", "AWS"], min_years=3.0)
        ranked = rank_candidates([weak, strong], jd)
        assert ranked[0].full_name == "Strong"
        assert ranked[0].ranking.rank == 1
        assert ranked[1].ranking.rank == 2

    def test_populates_insights_for_every_candidate(self):
        candidates = [make_candidate(full_name=f"C{i}", skills=["Python"],
                                       matching=MatchingResult(matched_required=["Python"])) for i in range(3)]
        jd = make_jd(required_skills=["Python"])
        ranked = rank_candidates(candidates, jd)
        assert all(c.insights.recruiter_summary for c in ranked)

    def test_empty_candidate_list(self):
        assert rank_candidates([], make_jd()) == []

    def test_ties_still_get_distinct_sequential_ranks(self):
        candidates = [make_candidate(full_name=f"C{i}", matching=MatchingResult()) for i in range(3)]
        jd = make_jd()
        ranked = rank_candidates(candidates, jd)
        assert [c.ranking.rank for c in ranked] == [1, 2, 3]


class TestScoringVersionAndReproducibility:
    def test_ranking_result_carries_scoring_version(self, engine):
        candidate = make_candidate(matching=MatchingResult(matched_required=["Python"]))
        jd = make_jd(required_skills=["Python"])
        result = engine.rank(candidate, jd)
        assert result.scoring_version == "ranking_v1"

    def test_ranking_result_carries_weights_snapshot(self, engine):
        candidate = make_candidate(matching=MatchingResult(matched_required=["Python"]))
        jd = make_jd(required_skills=["Python"])
        result = engine.rank(candidate, jd)
        assert result.weights_snapshot["skills"] == pytest.approx(0.35)
        assert sum(result.weights_snapshot.values()) == pytest.approx(1.0)


class TestRankingHistory:
    def test_first_ranking_does_not_create_history(self):
        candidate = make_candidate(matching=MatchingResult(matched_required=["Python"]))
        jd = make_jd(required_skills=["Python"])
        rank_candidates([candidate], jd)
        assert candidate.ranking_history == []

    def test_reranking_snapshots_the_previous_result(self):
        candidate = make_candidate(matching=MatchingResult(matched_required=["Python"]))
        jd = make_jd(required_skills=["Python"])
        rank_candidates([candidate], jd)
        first_score = candidate.ranking.overall_score

        rank_candidates([candidate], jd)  # re-rank the same candidate
        assert len(candidate.ranking_history) == 1
        assert candidate.ranking_history[0].overall_score == first_score


class TestBatchRankingStatistics:
    def test_empty_list_returns_zeroed_stats(self):
        from core.ranking import compute_batch_statistics as compute_ranking_stats
        stats = compute_ranking_stats([])
        assert stats.candidate_count == 0

    def test_computes_correct_aggregates(self):
        from core.ranking import compute_batch_statistics as compute_ranking_stats
        candidates = [
            make_candidate(full_name="A", matching=MatchingResult(matched_required=["Python"])),
            make_candidate(full_name="B", matching=MatchingResult()),
        ]
        jd = make_jd(required_skills=["Python"])
        ranked = rank_candidates(candidates, jd)
        stats = compute_ranking_stats(ranked)
        assert stats.candidate_count == 2
        assert stats.highest_score >= stats.average_score >= stats.lowest_score
        assert stats.highest_score >= stats.median_score >= stats.lowest_score

    def test_single_candidate_has_zero_stdev(self):
        from core.ranking import compute_batch_statistics as compute_ranking_stats
        candidate = make_candidate(matching=MatchingResult(matched_required=["Python"]))
        jd = make_jd(required_skills=["Python"])
        ranked = rank_candidates([candidate], jd)
        stats = compute_ranking_stats(ranked)
        assert stats.std_deviation == 0.0

    def test_recommendation_counts_tally_correctly(self):
        from core.ranking import compute_batch_statistics as compute_ranking_stats
        strong = make_candidate(full_name="Strong", skills=["Python"], total_years=6.0,
                                 matching=MatchingResult(matched_required=["Python"], similarity_score=90.0))
        weak = make_candidate(full_name="Weak", matching=MatchingResult(missing_required=["Python"]))
        jd = make_jd(required_skills=["Python"], min_years=3.0)
        ranked = rank_candidates([strong, weak], jd)
        stats = compute_ranking_stats(ranked)
        assert sum(stats.recommendation_counts.values()) == 2


class TestEnrichedRecruiterSummary:
    def test_summary_mentions_strong_categories(self, engine, generator=None):
        from core.ranking import InsightsGenerator
        generator = InsightsGenerator()
        candidate = make_candidate(
            skills=["Python", "AWS"], total_years=6.0, education=[Education(degree_level="Master")],
            matching=MatchingResult(matched_required=["Python", "AWS"], similarity_score=90.0),
        )
        jd = make_jd(required_skills=["Python", "AWS"], min_years=3.0)
        candidate.ranking = engine.rank(candidate, jd)
        insights = generator.generate(candidate)
        assert "Strong on" in insights.recruiter_summary or "Strongest" in insights.recruiter_summary
