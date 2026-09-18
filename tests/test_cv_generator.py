"""Unit tests for core/cv_generator.py  includes a full round-trip through
the real parser/extractor to confirm generated CVs are actually parseable,
not just valid PDFs.
"""
import pytest

from core.cv_generator import SAMPLE_PROFILES, generate_sample_resume_pdf, list_available_profiles
from core.extractor import ResumeExtractor
from core.parser import ResumeParser


class TestListAvailableProfiles:
    def test_returns_all_profiles(self):
        profiles = list_available_profiles()
        assert len(profiles) == len(SAMPLE_PROFILES)

    def test_each_entry_has_key_and_label(self):
        for profile in list_available_profiles():
            assert "key" in profile and "label" in profile
            assert profile["key"] and profile["label"]

    def test_includes_assignment_specified_archetypes(self):
        keys = {p["key"] for p in list_available_profiles()}
        assert "ai_ml_engineer" in keys
        assert "full_stack_developer" in keys


class TestGenerateSampleResumePdf:
    def test_raises_on_unknown_profile(self):
        with pytest.raises(ValueError, match="Unknown sample profile"):
            generate_sample_resume_pdf("not_a_real_profile")

    @pytest.mark.parametrize("profile_key", list(SAMPLE_PROFILES.keys()))
    def test_generates_nonempty_pdf_bytes(self, profile_key):
        pdf_bytes = generate_sample_resume_pdf(profile_key)
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 500


class TestGeneratedCvsAreActuallyParseable:
    """The real test: generated CVs must survive a round trip through the
    actual parser + extractor with high confidence, not just be valid PDFs.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def parser():
        return ResumeParser()

    @pytest.fixture(scope="class")
    @staticmethod
    def extractor():
        return ResumeExtractor()

    @pytest.mark.parametrize("profile_key", list(SAMPLE_PROFILES.keys()))
    def test_round_trip_extracts_correct_name(self, parser, extractor, profile_key):
        profile = SAMPLE_PROFILES[profile_key]
        pdf_bytes = generate_sample_resume_pdf(profile_key)
        doc = parser.parse(pdf_bytes, f"{profile_key}.pdf")
        candidate = extractor.extract(doc)
        assert candidate.full_name == profile.full_name

    @pytest.mark.parametrize("profile_key", list(SAMPLE_PROFILES.keys()))
    def test_round_trip_extracts_email_and_phone(self, parser, extractor, profile_key):
        profile = SAMPLE_PROFILES[profile_key]
        pdf_bytes = generate_sample_resume_pdf(profile_key)
        doc = parser.parse(pdf_bytes, f"{profile_key}.pdf")
        candidate = extractor.extract(doc)
        assert candidate.email == profile.email
        assert candidate.phone is not None

    @pytest.mark.parametrize("profile_key", list(SAMPLE_PROFILES.keys()))
    def test_round_trip_detects_most_listed_skills(self, parser, extractor, profile_key):
        profile = SAMPLE_PROFILES[profile_key]
        pdf_bytes = generate_sample_resume_pdf(profile_key)
        doc = parser.parse(pdf_bytes, f"{profile_key}.pdf")
        candidate = extractor.extract(doc)
        # Not every listed skill is necessarily in the taxonomy verbatim,
        # but the large majority should be detected.
        detected = set(candidate.skills)
        overlap = detected & set(profile.skills)
        assert len(overlap) >= len(profile.skills) * 0.6

    @pytest.mark.parametrize("profile_key", list(SAMPLE_PROFILES.keys()))
    def test_round_trip_yields_high_parsing_confidence(self, parser, extractor, profile_key):
        pdf_bytes = generate_sample_resume_pdf(profile_key)
        doc = parser.parse(pdf_bytes, f"{profile_key}.pdf")
        candidate = extractor.extract(doc)
        assert candidate.diagnostics.confidence >= 80.0

    @pytest.mark.parametrize("profile_key", list(SAMPLE_PROFILES.keys()))
    def test_round_trip_extracts_experience_and_education(self, parser, extractor, profile_key):
        pdf_bytes = generate_sample_resume_pdf(profile_key)
        doc = parser.parse(pdf_bytes, f"{profile_key}.pdf")
        candidate = extractor.extract(doc)
        assert len(candidate.experience) >= 1
        assert len(candidate.education) >= 1
        assert candidate.total_years_experience > 0
