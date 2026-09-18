"""Unit tests for models/candidate.py  property logic and serialization,
not extraction (that's tested in test_extractor.py once Module 2 lands).
"""
from datetime import datetime

from models.candidate import (
    AIInsights,
    Candidate,
    ExperienceEntry,
    MatchingResult,
    ParsingDiagnostics,
    ProcessingStatus,
    ProjectEntry,
    RankingResult,
    ResumeMetadata,
)


class TestCandidateIdentity:
    def test_display_name_prefers_full_name(self):
        c = Candidate(full_name="Jane Doe", metadata=ResumeMetadata(filename="resume.pdf"))
        assert c.display_name == "Jane Doe"

    def test_display_name_falls_back_to_filename(self):
        c = Candidate(full_name=None, metadata=ResumeMetadata(filename="john_resume.pdf"))
        assert c.display_name == "john_resume.pdf"

    def test_display_name_falls_back_to_unknown(self):
        c = Candidate(full_name=None, metadata=ResumeMetadata(filename=""))
        assert c.display_name == "Unknown Candidate"

    def test_candidate_ids_are_unique(self):
        ids = {Candidate().candidate_id for _ in range(100)}
        assert len(ids) == 100


class TestCandidateComputedProperties:
    def test_skill_count(self):
        c = Candidate(skills=["Python", "AWS", "Docker"])
        assert c.skill_count == 3

    def test_has_deployed_project_true_when_github_link_present(self):
        c = Candidate(projects=[ProjectEntry(name="X", github_link="https://github.com/x/y")])
        assert c.has_deployed_project is True

    def test_has_deployed_project_false_when_no_links(self):
        c = Candidate(projects=[ProjectEntry(name="X")])
        assert c.has_deployed_project is False


class TestExperienceEntryDuration:
    def test_duration_years_for_completed_role(self):
        e = ExperienceEntry(start_year=2018, end_year=2021, is_current=False)
        assert e.duration_years == 3.0

    def test_duration_years_for_current_role_uses_present_year(self):
        e = ExperienceEntry(start_year=2020, is_current=True)
        assert e.duration_years == float(datetime.now().year - 2020)

    def test_duration_years_zero_when_no_start_year(self):
        e = ExperienceEntry()
        assert e.duration_years == 0.0


class TestParsingDiagnostics:
    def test_default_status_is_uploaded(self):
        c = Candidate()
        assert c.diagnostics.status == ProcessingStatus.UPLOADED

    def test_status_progresses_through_pipeline_stages(self):
        c = Candidate()
        for stage in (
            ProcessingStatus.VALIDATED, ProcessingStatus.PARSED, ProcessingStatus.EXTRACTED,
            ProcessingStatus.RANKED, ProcessingStatus.COMPLETED,
        ):
            c.diagnostics.status = stage
        assert c.diagnostics.status == ProcessingStatus.COMPLETED

    def test_add_warning_appends(self):
        diag = ParsingDiagnostics()
        diag.add_warning("Email missing")
        diag.add_warning("Phone not found")
        assert diag.warnings == ["Email missing", "Phone not found"]

    def test_confidence_defaults_to_none_until_extractor_sets_it(self):
        assert Candidate().diagnostics.confidence is None

    def test_low_confidence_and_warnings_surface_in_chat_context(self):
        c = Candidate(full_name="Jane Doe")
        c.diagnostics.confidence = 62.0
        c.diagnostics.add_warning("Education section not found")
        summary = c.to_summary_dict()
        assert summary["parser_confidence"] == 62.0
        assert "Education section not found" in summary["parsing_warnings"]


class TestResumeMetadataTiming:
    def test_per_stage_timings_default_to_zero(self):
        meta = ResumeMetadata()
        assert meta.parse_time_ms == meta.extract_time_ms == meta.rank_time_ms == 0.0

    def test_per_stage_timings_are_independently_settable(self):
        meta = ResumeMetadata(parse_time_ms=120.5, extract_time_ms=340.2, rank_time_ms=15.1)
        meta.processing_time_ms = meta.parse_time_ms + meta.extract_time_ms + meta.rank_time_ms
        assert meta.processing_time_ms == 120.5 + 340.2 + 15.1

    def test_timing_included_in_json_export(self):
        c = Candidate(metadata=ResumeMetadata(filename="x.pdf", parse_time_ms=50.0, extract_time_ms=100.0))
        data = c.to_json()
        assert data["metadata"]["timing_ms"]["parse"] == 50.0
        assert data["metadata"]["timing_ms"]["extract"] == 100.0


class TestMatchingResult:
    def test_defaults(self):
        m = MatchingResult()
        assert m.similarity_score == 0.0
        assert m.matched_skills == []
        assert m.confidence is None

    def test_to_json_includes_full_matching_object(self):
        c = Candidate(full_name="Jane Doe")
        c.matching = MatchingResult(
            similarity_score=72.5, matched_skills=["Python"], missing_skills=["Go"],
            matched_keywords=["cloud infrastructure"], explanation="Strong Python match.",
        )
        data = c.to_json()
        assert data["matching"]["similarity_score"] == 72.5
        assert data["matching"]["matched_keywords"] == ["cloud infrastructure"]
        assert data["matching"]["explanation"] == "Strong Python match."


class TestCandidateSerialization:
    def _sample_candidate(self) -> Candidate:
        c = Candidate(
            full_name="Jane Doe",
            email="jane@example.com",
            skills=["Python", "AWS"],
            metadata=ResumeMetadata(filename="jane.pdf"),
        )
        c.matching = MatchingResult(matched_skills=["Python"], missing_skills=["Kubernetes"])
        c.ranking = RankingResult(overall_score=87.456, skill_score=90.0, rank=1, recommendation="Highly Recommended")
        c.insights = AIInsights(strengths=["Strong Python background"], weaknesses=["No Kubernetes experience"])
        return c

    def test_to_summary_dict_rounds_scores_and_excludes_raw_text(self):
        c = self._sample_candidate()
        c.raw_text = "this should never leak into chatbot context " * 50
        summary = c.to_summary_dict()
        assert "raw_text" not in summary
        assert summary["overall_score"] == 87.5
        assert summary["rank"] == 1
        assert summary["name"] == "Jane Doe"

    def test_to_export_row_is_flat_and_spreadsheet_ready(self):
        c = self._sample_candidate()
        row = c.to_export_row()
        assert row["Name"] == "Jane Doe"
        assert row["Matched Skills"] == "Python"
        assert isinstance(row["Overall Score"], float)

    def test_to_json_excludes_raw_text_but_keeps_structured_fields(self):
        c = self._sample_candidate()
        c.raw_text = "full resume body text"
        data = c.to_json()
        assert "raw_text" not in data
        assert data["contact"]["email"] == "jane@example.com"
        assert data["ranking"]["overall_score"] == 87.456
