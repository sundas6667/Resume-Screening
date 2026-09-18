"""Unit tests for utils/validators.py  the input-boundary guards."""
from config import MAX_FILE_SIZE_MB
from utils.validators import (
    is_duplicate_content,
    sanitize_text,
    validate_job_description,
    validate_pdf_content,
    validate_uploaded_file,
)


class TestValidateUploadedFile:
    def test_accepts_valid_pdf(self):
        result = validate_uploaded_file("resume.pdf", 1024)
        assert result.is_valid

    def test_rejects_non_pdf_extension(self):
        result = validate_uploaded_file("resume.docx", 1024)
        assert not result.is_valid
        assert "not a supported format" in result.message

    def test_rejects_empty_filename(self):
        result = validate_uploaded_file("", 1024)
        assert not result.is_valid

    def test_rejects_zero_byte_file(self):
        result = validate_uploaded_file("resume.pdf", 0)
        assert not result.is_valid

    def test_rejects_oversized_file(self):
        too_big = (MAX_FILE_SIZE_MB + 1) * 1024 * 1024
        result = validate_uploaded_file("resume.pdf", too_big)
        assert not result.is_valid
        assert "size limit" in result.message

    def test_is_case_insensitive_on_extension(self):
        result = validate_uploaded_file("Resume.PDF", 1024)
        assert result.is_valid


class TestValidateJobDescription:
    def test_rejects_empty_text(self):
        assert not validate_job_description("").is_valid
        assert not validate_job_description(None).is_valid

    def test_rejects_too_short_text(self):
        assert not validate_job_description("Python developer needed").is_valid

    def test_accepts_sufficiently_detailed_text(self):
        jd = "We are hiring a Senior AI/ML Engineer with 5+ years of Python, " \
             "TensorFlow, NLP, AWS, and Docker experience for our platform team."
        assert validate_job_description(jd).is_valid


class TestValidatePdfContent:
    def test_rejects_encrypted_pdf(self):
        result = validate_pdf_content(is_encrypted=True, page_count=2, extracted_char_count=1000)
        assert not result.is_valid
        assert "password-protected" in result.message

    def test_rejects_zero_page_pdf(self):
        result = validate_pdf_content(is_encrypted=False, page_count=0, extracted_char_count=0)
        assert not result.is_valid

    def test_rejects_likely_scanned_pdf(self):
        result = validate_pdf_content(is_encrypted=False, page_count=1, extracted_char_count=10)
        assert not result.is_valid
        assert "scanned" in result.message

    def test_accepts_normal_pdf(self):
        result = validate_pdf_content(is_encrypted=False, page_count=2, extracted_char_count=2500)
        assert result.is_valid


class TestDuplicateDetection:
    def test_detects_duplicate_hash(self):
        seen = {"abc123"}
        assert is_duplicate_content("abc123", seen) is True

    def test_does_not_flag_new_hash(self):
        seen = {"abc123"}
        assert is_duplicate_content("xyz789", seen) is False


class TestSanitizeText:
    def test_strips_null_bytes(self):
        assert "\x00" not in sanitize_text("hello\x00world")

    def test_truncates_overly_long_text(self):
        long_text = "a" * 1000
        result = sanitize_text(long_text, max_length=100)
        assert len(result) == 100

    def test_handles_empty_string(self):
        assert sanitize_text("") == ""
