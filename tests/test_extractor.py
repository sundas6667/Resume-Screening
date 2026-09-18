"""Unit tests for core/extractor.py  SkillDatabase detection accuracy and
ResumeExtractor's structured extraction, including regression tests for
bugs found during manual integration testing (see inline comments).
"""
import pytest

from core.extractor import ResumeExtractor, SkillDatabase, get_skill_database
from core.parser import ParsedDocument
from utils.helpers import clean_text


@pytest.fixture(scope="module")
def skill_db() -> SkillDatabase:
    return get_skill_database()


@pytest.fixture(scope="module")
def extractor() -> ResumeExtractor:
    return ResumeExtractor()


def make_doc(text: str, filename: str = "test.pdf") -> ParsedDocument:
    cleaned = clean_text(text)
    return ParsedDocument(
        filename=filename, raw_text=text, cleaned_text=cleaned,
        page_count=1, file_size_bytes=len(text.encode()), content_hash="testhash",
    )


# --------------------------------------------------------------------------- #
# SkillDatabase
# --------------------------------------------------------------------------- #
class TestSkillDatabaseBasics:
    def test_loads_over_200_skills(self, skill_db):
        assert len(skill_db.all_skills) > 200

    def test_detects_exact_canonical_name(self, skill_db):
        assert "Python" in skill_db.detect("I write Python code daily.")

    def test_detects_known_alias(self, skill_db):
        assert "JavaScript" in skill_db.detect("Strong background in JS.")
        assert "Kubernetes" in skill_db.detect("Deployed with K8s.")

    def test_returns_empty_list_for_no_matches(self, skill_db):
        assert skill_db.detect("The quick brown fox jumps over the lazy dog.") == []

    def test_returns_empty_list_for_empty_text(self, skill_db):
        assert skill_db.detect("") == []

    def test_category_lookup(self, skill_db):
        assert skill_db.category_of("Python") == "Programming Languages"
        assert skill_db.category_of("NotARealSkill") is None


class TestSkillDatabaseSpacingTolerance:
    """Point 4 from the review: 'TensorFlow' / 'Tensor Flow' / 'tensor-flow' should all match."""

    @pytest.mark.parametrize("variant", ["TensorFlow", "Tensor Flow", "tensor-flow", "TENSORFLOW", "tensor_flow"])
    def test_tensorflow_spacing_variants(self, skill_db, variant):
        assert "TensorFlow" in skill_db.detect(f"Experience with {variant} models.")


class TestSkillDatabaseFalsePositiveGuards:
    """Regression tests: substring matches inside unrelated words must NOT fire."""

    def test_r_language_does_not_match_inside_hr(self, skill_db):
        assert "R" not in skill_db.detect("I am a Senior HR Manager.")

    def test_r_language_does_not_match_inside_performance(self, skill_db):
        assert "R" not in skill_db.detect("We improved performance for our customers.")

    def test_r_language_matches_as_standalone_word(self, skill_db):
        assert "R" in skill_db.detect("Proficient in R for statistical analysis.")

    def test_go_does_not_match_inside_google(self, skill_db):
        assert "Go" not in skill_db.detect("Built scalable systems at Google.")

    def test_go_matches_as_standalone_word(self, skill_db):
        assert "Go" in skill_db.detect("Backend services written in Go.")

    def test_c_plus_plus_and_bare_c_both_detected_correctly(self, skill_db):
        found = skill_db.detect("Experience with C++, C#, and C.")
        assert {"C++", "C#", "C"}.issubset(found)

    def test_longer_term_preferred_over_shorter_prefix(self, skill_db):
        # "React Native" must be detected distinctly from bare "React"
        found = skill_db.detect("Built apps with React and React Native.")
        assert "React" in found
        assert "React Native" in found


# --------------------------------------------------------------------------- #
# Contact info extraction
# --------------------------------------------------------------------------- #
class TestContactExtraction:
    def test_extracts_email(self, extractor):
        c = extractor.extract(make_doc("Jane Doe\njane.doe@example.com\nSKILLS\nPython"))
        assert c.email == "jane.doe@example.com"

    def test_extracts_phone_from_header(self, extractor):
        c = extractor.extract(make_doc("Jane Doe\n+1 415-555-0199\nSKILLS\nPython, Docker, AWS, Kubernetes"))
        assert c.phone is not None
        assert sum(ch.isdigit() for ch in c.phone) >= 7

    def test_extracts_linkedin_and_github(self, extractor):
        text = "Jane Doe\nlinkedin.com/in/janedoe | github.com/janedoe\nSKILLS\nPython"
        c = extractor.extract(make_doc(text))
        assert "linkedin.com/in/janedoe" in c.linkedin
        assert "github.com/janedoe" in c.github

    def test_extracts_name_from_first_line(self, extractor):
        c = extractor.extract(make_doc("Jane Doe\njane@example.com\nSKILLS\nPython"))
        assert c.full_name == "Jane Doe"

    def test_skips_resume_title_word_for_name(self, extractor):
        text = "RESUME\nJane Doe\njane@example.com\nSKILLS\nPython"
        c = extractor.extract(make_doc(text))
        assert c.full_name == "Jane Doe"

    def test_falls_back_to_none_when_name_undetectable(self, extractor):
        # No plausible name-shaped line in the first few lines
        text = "jane@example.com\n123456\nSKILLS\nPython"
        c = extractor.extract(make_doc(text))
        assert c.full_name is None  # Candidate.display_name will fall back to filename


class TestLocationExtraction:
    """Regression tests for the 'Python, TensorFlow' mistaken-for-location bug."""

    def test_extracts_city_state_from_contact_line(self, extractor):
        text = "Jane Doe\njane@example.com | +1 415-555-0199 | San Francisco, CA\n\nSKILLS\nPython, AWS"
        c = extractor.extract(make_doc(text))
        assert c.location == "San Francisco, CA"

    def test_does_not_mistake_skills_list_for_location(self, extractor):
        text = "Jane Doe\njane@example.com\n\nSKILLS\nPython, TensorFlow, PyTorch, AWS, Docker"
        c = extractor.extract(make_doc(text))
        assert c.location is None

    def test_returns_none_when_no_location_present(self, extractor):
        text = "Jane Doe\njane@example.com\n\nSKILLS\nPython"
        c = extractor.extract(make_doc(text))
        assert c.location is None


# --------------------------------------------------------------------------- #
# Section splitting
# --------------------------------------------------------------------------- #
class TestSectionSplitting:
    def test_standard_headings(self, extractor):
        text = (
            "Jane Doe\njane@example.com\n\n"
            "SUMMARY\nExperienced engineer.\n\n"
            "SKILLS\nPython, AWS\n\n"
            "EXPERIENCE\nSoftware Engineer, Acme Corp   2020 - Present\n- Built things\n\n"
            "EDUCATION\nBachelor of Science in Computer Science\nMIT, 2019"
        )
        sections = extractor._split_into_sections(clean_text(text))
        assert "summary" in sections and "Experienced engineer" in sections["summary"]
        assert "skills" in sections
        assert "experience" in sections
        assert "education" in sections

    @pytest.mark.parametrize("heading,category", [
        ("EMPLOYMENT", "experience"),
        ("CAREER", "experience"),
        ("PROFESSIONAL HISTORY", "experience"),
        ("QUALIFICATIONS", "education"),
        ("ACADEMICS", "education"),
        ("PORTFOLIO", "projects"),
        ("CAREER OBJECTIVE", "summary"),
    ])
    def test_nonstandard_headings_map_to_correct_category(self, extractor, heading, category):
        text = f"Jane Doe\njane@example.com\n\n{heading}\nSome relevant content here.\n\nSKILLS\nPython"
        sections = extractor._split_into_sections(clean_text(text))
        assert category in sections
        assert "Some relevant content" in sections[category]

    def test_bare_career_does_not_collide_with_career_objective(self, extractor):
        text = (
            "Jane Doe\njane@example.com\n\n"
            "CAREER OBJECTIVE\nSeeking a challenging role.\n\n"
            "CAREER\nSoftware Engineer, Acme   2020 - Present"
        )
        sections = extractor._split_into_sections(clean_text(text))
        assert "Seeking a challenging role" in sections["summary"]
        assert "Software Engineer" in sections["experience"]

    def test_multiple_headings_for_same_category_are_merged(self, extractor):
        text = (
            "Jane Doe\njane@example.com\n\n"
            "EMPLOYMENT\nRole A, Company A   2022 - Present\n\n"
            "PROFESSIONAL HISTORY\nRole B, Company B   2019 - 2022"
        )
        sections = extractor._split_into_sections(clean_text(text))
        assert "Role A" in sections["experience"]
        assert "Role B" in sections["experience"]


# --------------------------------------------------------------------------- #
# Education
# --------------------------------------------------------------------------- #
class TestEducationExtraction:
    def test_extracts_degree_level_and_field(self, extractor):
        text = "Master of Science in Computer Science\nStanford University, 2020\nGPA: 3.9/4.0"
        edu = extractor._extract_education(text)
        assert len(edu) == 1
        assert edu[0].degree_level == "Master"
        assert edu[0].field_of_study == "Computer Science"
        assert edu[0].graduation_year == 2020
        assert "3.9" in edu[0].cgpa

    def test_extracts_known_university(self, extractor):
        text = "Bachelor of Science in Computer Science\nMassachusetts Institute of Technology, 2018"
        edu = extractor._extract_education(text)
        assert edu[0].institution == "Massachusetts Institute of Technology"

    def test_extracts_honors(self, extractor):
        text = "Bachelor of Science in Computer Science, magna cum laude\nUniversity of Washington, 2021"
        edu = extractor._extract_education(text)
        assert edu[0].honors is not None
        assert "magna cum laude" in edu[0].honors.lower()

    def test_empty_section_returns_empty_list(self, extractor):
        assert extractor._extract_education("") == []

    def test_multiple_degrees_split_correctly(self, extractor):
        text = (
            "Master of Science in Computer Science\nStanford University, 2020\n\n"
            "Bachelor of Science in Mathematics\nUC Berkeley, 2018"
        )
        edu = extractor._extract_education(text)
        assert len(edu) == 2
        levels = {e.degree_level for e in edu}
        assert levels == {"Master", "Bachelor"}


class TestDegreeAbbreviationBoundaries:
    """Regression tests: 'M.E.'/'B.A.'-style abbreviations previously matched
    mid-word (e.g. 'require-ME-nts') without \\b boundaries.
    """

    def test_short_abbreviation_does_not_match_inside_ordinary_word(self, extractor):
        level = extractor._extract_required_education_level(
            "Requirements: 5+ years of Python and machine learning experience."
        )
        assert level is None

    def test_bare_bachelor_word_is_detected(self, extractor):
        assert extractor._extract_required_education_level(
            "We require a Bachelor Degree in Computer Science."
        ) == "Bachelor"

    def test_bare_masters_word_is_detected(self, extractor):
        assert extractor._extract_required_education_level(
            "Master's degree in a related field required."
        ) == "Master"

    def test_highest_rank_wins_when_multiple_levels_mentioned(self, extractor):
        assert extractor._extract_required_education_level(
            "Master's degree or PhD in Computer Science required."
        ) == "PhD"


# --------------------------------------------------------------------------- #
# Experience
# --------------------------------------------------------------------------- #
class TestExperienceExtraction:
    def test_extracts_title_company_and_years(self, extractor):
        text = "Senior Software Engineer, Acme Corp   2019 - 2023\n- Led backend development"
        entries, total_years, seniority = extractor._extract_experience(text)
        assert len(entries) == 1
        assert entries[0].designation == "Senior Software Engineer"
        assert entries[0].company == "Acme Corp"
        assert entries[0].start_year == 2019
        assert entries[0].end_year == 2023
        assert entries[0].is_current is False

    def test_detects_current_role(self, extractor):
        text = "Software Engineer, Acme Corp   2021 - Present"
        entries, _, _ = extractor._extract_experience(text)
        assert entries[0].is_current is True
        assert entries[0].end_year is None

    def test_detects_internship(self, extractor):
        text = "Software Engineering Intern, Acme Corp   2022 - 2023"
        entries, _, _ = extractor._extract_experience(text)
        assert entries[0].is_internship is True

    def test_multiple_roles_produce_multiple_entries(self, extractor):
        text = (
            "Senior Engineer, Acme Corp   2021 - Present\n- Led team\n"
            "Engineer, Beta Inc   2018 - 2021\n- Built features"
        )
        entries, total_years, _ = extractor._extract_experience(text)
        assert len(entries) == 2
        assert total_years > 0

    def test_empty_section_returns_no_experience(self, extractor):
        entries, total_years, seniority = extractor._extract_experience("")
        assert entries == []
        assert total_years == 0.0
        assert seniority is None

    def test_entries_sorted_newest_first_regardless_of_source_order(self, extractor):
        # Deliberately listed OLDEST first in the source text.
        text = (
            "Engineer, OldCo   2015 - 2018\n- Did early-career work\n"
            "Senior Engineer, NewCo   2021 - Present\n- Leads the platform team\n"
            "Engineer, MidCo   2018 - 2021\n- Built core features"
        )
        entries, _, _ = extractor._extract_experience(text)
        assert len(entries) == 3
        assert entries[0].is_current is True
        assert entries[0].company == "NewCo"
        assert entries[1].start_year == 2018
        assert entries[2].start_year == 2015


class TestSeniorityInference:
    """Regression tests for two bugs: substring false-positives ('architect'
    inside 'architecture') and stale historical titles outranking current role.
    """

    def test_architecture_does_not_trigger_architect_seniority(self, extractor):
        text = "Full Stack Developer, StartupXYZ   2020 - Present\n- Managed microservices architecture"
        _, _, seniority = extractor._extract_experience(text)
        assert seniority != "Architect"

    def test_current_role_keyword_takes_priority(self, extractor):
        text = "Senior Machine Learning Engineer, DeepMind   2021 - Present\n- Led a team"
        _, _, seniority = extractor._extract_experience(text)
        assert seniority == "Senior"

    def test_old_junior_title_does_not_outrank_years_of_subsequent_experience(self, extractor):
        text = (
            "Full Stack Developer, StartupXYZ   2020 - Present\n- Managed microservices\n"
            "Junior Developer, WebAgency   2019 - 2020\n- Built WordPress sites"
        )
        entries, total_years, seniority = extractor._extract_experience(text)
        assert total_years >= 5
        assert seniority != "Junior"

    def test_new_grad_with_no_keyword_and_low_years_is_junior(self, extractor):
        text = "Software Developer, SmallCo   2025 - Present\n- Fixed bugs"
        _, total_years, seniority = extractor._extract_experience(text)
        if total_years < 2:
            assert seniority == "Junior"


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
class TestProjectExtraction:
    def test_extracts_name_description_and_tech_stack(self, extractor):
        text = "Real-time Chat App\nBuilt a chat application using Python and WebSockets."
        projects = extractor._extract_projects(text)
        assert len(projects) == 1
        assert projects[0].name == "Real-time Chat App"
        assert "Python" in projects[0].tech_stack

    def test_extracts_github_link(self, extractor):
        text = "Chat App\nA real time chat app.\ngithub.com/jane/chat-app"
        projects = extractor._extract_projects(text)
        assert projects[0].github_link is not None
        assert "github.com/jane/chat-app" in projects[0].github_link

    def test_github_link_line_does_not_become_its_own_fake_project(self, extractor):
        """Regression test: a bare URL line was previously mistaken for a new
        project title, splitting one project into two."""
        text = (
            "Real-time Object Detection System\n"
            "Built a YOLO-based detection system deployed on edge devices.\n"
            "github.com/johnanderson/object-detection"
        )
        projects = extractor._extract_projects(text)
        assert len(projects) == 1
        assert projects[0].name == "Real-time Object Detection System"

    def test_empty_section_returns_empty_list(self, extractor):
        assert extractor._extract_projects("") == []

    def test_multiple_projects_split_on_blank_lines(self, extractor):
        text = (
            "Project One\nA description of project one using Python.\n\n"
            "Project Two\nA description of project two using React."
        )
        projects = extractor._extract_projects(text)
        assert len(projects) == 2


# --------------------------------------------------------------------------- #
# Certifications
# --------------------------------------------------------------------------- #
class TestCertificationExtraction:
    def test_extracts_explicit_certification_lines(self, extractor):
        text = "AWS Certified Solutions Architect\nGoogle Professional Data Engineer"
        certs = extractor._extract_certifications(text, text)
        assert "AWS Certified Solutions Architect" in certs
        assert "Google Professional Data Engineer" in certs

    def test_does_not_duplicate_provider_already_covered_by_explicit_line(self, extractor):
        """Regression test: a generic 'AWS Certification' fallback was
        previously appended even when an explicit AWS cert line already existed."""
        text = "AWS Certified Solutions Architect"
        certs = extractor._extract_certifications(text, text)
        assert certs.count("AWS Certified Solutions Architect") == 1
        assert not any(c == "AWS Certification" for c in certs)

    def test_no_certifications_section_and_no_mentions_returns_empty(self, extractor):
        assert extractor._extract_certifications("", "I like hiking and reading books.") == []

    def test_detects_provider_mentioned_inline_without_dedicated_section(self, extractor):
        full_text = "Skilled engineer, AWS Certified Developer with cloud experience."
        certs = extractor._extract_certifications("", full_text)
        assert any("AWS" in c for c in certs)


# --------------------------------------------------------------------------- #
# Job description extraction
# --------------------------------------------------------------------------- #
class TestJobDescriptionExtraction:
    SAMPLE_JD = (
        "Senior AI/ML Engineer\n"
        "Requirements: 5+ years Python and ML experience, TensorFlow/PyTorch, NLP, "
        "Computer Vision, AWS/GCP, Docker, Kubernetes, MLOps. Preferred: RAG systems, "
        "FastAPI, PostgreSQL, Team leadership."
    )

    def test_extracts_title(self, extractor):
        jd = extractor.extract_job_description(self.SAMPLE_JD)
        assert jd.title == "Senior AI/ML Engineer"

    def test_extracts_required_skills(self, extractor):
        jd = extractor.extract_job_description(self.SAMPLE_JD)
        assert "Python" in jd.required_skills
        assert "TensorFlow" in jd.required_skills
        assert "Kubernetes" in jd.required_skills

    def test_preferred_skills_excluded_from_required(self, extractor):
        jd = extractor.extract_job_description(self.SAMPLE_JD)
        assert "FastAPI" in jd.preferred_skills
        assert "FastAPI" not in jd.required_skills

    def test_extracts_min_years_experience(self, extractor):
        jd = extractor.extract_job_description(self.SAMPLE_JD)
        assert jd.min_years_experience == 5.0

    def test_no_degree_mentioned_returns_none(self, extractor):
        jd = extractor.extract_job_description(self.SAMPLE_JD)
        assert jd.required_education_level is None

    def test_all_relevant_skills_deduplicates(self, extractor):
        jd = extractor.extract_job_description(self.SAMPLE_JD)
        assert len(jd.all_relevant_skills) == len(set(jd.all_relevant_skills))

    def test_is_ready_true_for_valid_jd(self, extractor):
        jd = extractor.extract_job_description(self.SAMPLE_JD)
        assert jd.is_ready is True


# --------------------------------------------------------------------------- #
# Diagnostics: per-field confidence and script detection
# --------------------------------------------------------------------------- #
class TestFieldConfidence:
    def test_all_fields_present_yields_full_confidence(self, extractor):
        text = (
            "Jane Doe\njane@example.com | +1 415-555-0199\n\n"
            "SUMMARY\nExperienced engineer.\n\n"
            "SKILLS\nPython, AWS\n\n"
            "EXPERIENCE\nEngineer, Acme   2020 - Present\n- Did things\n\n"
            "EDUCATION\nBachelor of Science in Computer Science\nMIT, 2019"
        )
        c = extractor.extract(make_doc(text))
        assert c.diagnostics.confidence == 100.0
        assert c.diagnostics.field_confidence["email"] == 100.0
        assert c.diagnostics.field_confidence["full_name"] == 100.0

    def test_missing_fields_reflected_in_breakdown_and_aggregate(self, extractor):
        text = "jane@example.com\n\nSKILLS\nPython"
        c = extractor.extract(make_doc(text))
        assert c.diagnostics.field_confidence["full_name"] == 0.0
        assert c.diagnostics.field_confidence["phone"] == 0.0
        assert c.diagnostics.confidence < 100.0

    def test_field_confidence_keys_match_weight_categories(self, extractor):
        c = extractor.extract(make_doc("Jane Doe\njane@example.com\n\nSKILLS\nPython"))
        assert set(c.diagnostics.field_confidence.keys()) == {
            "full_name", "email", "phone", "skills", "education", "experience", "summary",
        }


class TestScriptDetection:
    def test_english_resume_detected_as_latin(self, extractor):
        c = extractor.extract(make_doc("Jane Doe\njane@example.com\n\nSKILLS\nPython, AWS"))
        assert c.diagnostics.detected_script == "Latin"
        assert not any("script" in w.lower() for w in c.diagnostics.warnings)

    def test_arabic_script_text_detected_and_warned(self, extractor):
        # Enough Arabic-script characters to clear the detection threshold.
        arabic_text = "مرحبا بكم في هذه السيرة الذاتية الخاصة بي وهي تحتوي على معلومات مهمة جدا" * 2
        c = extractor.extract(make_doc(arabic_text))
        assert c.diagnostics.detected_script == "Arabic"
        assert any("script" in w.lower() for w in c.diagnostics.warnings)

    def test_empty_text_returns_unknown(self, extractor):
        assert extractor._detect_script("") == "Unknown"


class TestUnicodeNameExtraction:
    """Regression tests: name-shape validation was ASCII-only ([A-Za-z]),
    which rejected accented/non-Latin names even after clean_text() was
    fixed to stop stripping them."""

    def test_accented_latin_name_detected(self, extractor):
        c = extractor.extract(make_doc("José García\njose@example.com\n\nSKILLS\nPython"))
        assert c.full_name == "José García"

    def test_german_umlaut_name_detected(self, extractor):
        c = extractor.extract(make_doc("François Müller\nfm@example.com\n\nSKILLS\nPython"))
        assert c.full_name == "François Müller"

    def test_arabic_script_name_detected(self, extractor):
        c = extractor.extract(make_doc("محمد أحمد\nmohammed@example.com\n\nSKILLS\nPython"))
        assert c.full_name == "محمد أحمد"
