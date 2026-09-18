"""Unit tests for the regex patterns in utils/constants.py  these are the
foundation the extractor is built on, so they're worth pinning down early."""
from utils.constants import (
    EMAIL_PATTERN,
    GITHUB_PATTERN,
    LINKEDIN_PATTERN,
    PHONE_PATTERN,
    SECTION_HEADER_PATTERN,
    SECTION_HEADERS,
    SENIORITY_KEYWORDS,
    YEAR_RANGE_PATTERN,
)


class TestEmailPattern:
    def test_matches_standard_email(self):
        assert EMAIL_PATTERN.search("Contact: jane.doe@example.com for details")

    def test_matches_email_with_plus_tag(self):
        assert EMAIL_PATTERN.search("jane+resumes@example.co.uk")

    def test_does_not_match_bare_text(self):
        assert not EMAIL_PATTERN.search("no email address here")


class TestPhonePattern:
    def test_matches_international_format(self):
        assert PHONE_PATTERN.search("+1 (555) 123-4567")

    def test_matches_local_format(self):
        assert PHONE_PATTERN.search("0300-1234567")


class TestSocialLinkPatterns:
    def test_matches_linkedin_url(self):
        m = LINKEDIN_PATTERN.search("Profile: linkedin.com/in/jane-doe-123")
        assert m and "linkedin.com/in/jane-doe-123" in m.group()

    def test_matches_github_url(self):
        m = GITHUB_PATTERN.search("https://github.com/janedoe")
        assert m and "github.com/janedoe" in m.group()


class TestSectionHeaderPattern:
    def test_matches_known_headers_case_insensitively(self):
        for header in ("SKILLS", "Work Experience", "education", "Projects", "Certifications"):
            assert SECTION_HEADER_PATTERN.search(header), f"Expected match for header: {header}"

    def test_does_not_match_arbitrary_line(self):
        assert not SECTION_HEADER_PATTERN.search("I built a scalable microservice.")

    def test_loaded_from_json_not_hardcoded(self):
        # Real resumes use wildly inconsistent headings — this is the fix for that.
        assert "_note" not in SECTION_HEADERS
        assert set(SECTION_HEADERS.keys()) >= {"summary", "skills", "experience", "education", "projects"}

    def test_matches_bare_word_experience_variants(self):
        for header in ("Employment", "Career", "Professional History"):
            assert SECTION_HEADER_PATTERN.search(header), f"Expected match for: {header}"

    def test_bare_alias_does_not_bleed_into_a_different_category(self):
        # "Career" is a valid bare EXPERIENCE header, and "Career Objective" is a
        # separate, valid SUMMARY header — the full-line anchor must keep these
        # attributed to the correct category, not have "career" bleed across.
        line = "career objective"
        assert line in [h.lower() for h in SECTION_HEADERS["summary"]]
        assert line not in [h.lower() for h in SECTION_HEADERS["experience"]]
        assert "career" in [h.lower() for h in SECTION_HEADERS["experience"]]

    def test_matches_new_education_and_project_variants(self):
        for header in ("Qualifications", "Academics", "Portfolio"):
            assert SECTION_HEADER_PATTERN.search(header), f"Expected match for: {header}"


class TestYearRangePattern:
    def test_matches_year_to_year(self):
        m = YEAR_RANGE_PATTERN.search("Software Engineer, 2019 - 2022")
        assert m and m.group("start") == "2019"

    def test_matches_year_to_present(self):
        m = YEAR_RANGE_PATTERN.search("Senior Engineer, 2021 - Present")
        assert m and m.group("end").lower() == "present"


class TestSeniorityKeywords:
    def test_all_expected_levels_present(self):
        expected = {"Architect", "Manager", "Lead", "Senior", "Mid-Level", "Junior"}
        assert expected == set(SENIORITY_KEYWORDS.keys())

    def test_each_level_has_at_least_one_keyword(self):
        for level, keywords in SENIORITY_KEYWORDS.items():
            assert len(keywords) > 0, f"{level} has no keywords"
