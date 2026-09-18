"""
utils/validators.py
====================
Guards at the system boundary: every uploaded file and every piece of raw
text passes through here before it reaches the parser. Keeping validation
separate from parsing means the parser can assume its input is already safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import ALLOWED_RESUME_EXTENSIONS, MAX_FILE_SIZE_MB
from utils.helpers import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when an uploaded file or input fails validation.

    Always caught at the UI boundary and shown as a friendly message —
    never allowed to crash the app.
    """


@dataclass
class ValidationResult:
    is_valid: bool
    message: str = "OK"


def validate_uploaded_file(filename: str, file_size_bytes: int) -> ValidationResult:
    """Validate an uploaded resume before it's handed to the parser.

    Checks extension and size only — content-level validation (is it
    actually a readable PDF?) happens in core/parser.py, since that
    requires attempting the extraction itself.
    """
    if not filename:
        return ValidationResult(False, "File has no name.")

    lower_name = filename.lower()
    if not lower_name.endswith(ALLOWED_RESUME_EXTENSIONS):
        allowed = ", ".join(ALLOWED_RESUME_EXTENSIONS)
        return ValidationResult(False, f"'{filename}' is not a supported format. Allowed: {allowed}")

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes <= 0:
        return ValidationResult(False, f"'{filename}' is empty.")
    if file_size_bytes > max_bytes:
        return ValidationResult(False, f"'{filename}' exceeds the {MAX_FILE_SIZE_MB}MB size limit.")

    return ValidationResult(True)


def validate_job_description(text: Optional[str]) -> ValidationResult:
    """A job description that's too short produces meaningless TF-IDF
    vectors, so we require a minimal amount of real content.
    """
    if not text or not text.strip():
        return ValidationResult(False, "Job description is empty.")
    word_count = len(text.split())
    if word_count < 15:
        return ValidationResult(False, "Job description is too short to match against (add more detail).")
    return ValidationResult(True)


def validate_pdf_content(is_encrypted: bool, page_count: int, extracted_char_count: int) -> ValidationResult:
    """Validate a PDF *after* the parser has attempted to open it — this
    function stays free of any PyPDF2 dependency so it can be unit tested
    with plain booleans/ints; core/parser.py is what supplies them.
    """
    if is_encrypted:
        return ValidationResult(False, "PDF is password-protected. Please upload an unlocked file.")
    if page_count == 0:
        return ValidationResult(False, "PDF has no pages.")
    if extracted_char_count < 50:
        return ValidationResult(
            False,
            "Little to no text could be extracted — this looks like a scanned/image-based "
            "PDF, which isn't supported yet (OCR is on the roadmap).",
        )
    return ValidationResult(True)


def is_duplicate_content(content_hash: str, seen_hashes: set[str]) -> bool:
    """True if this exact file (by SHA-256) has already been processed in
    this session — used to silently skip re-processing the same resume
    uploaded twice, rather than double-counting it in rankings/analytics.
    """
    return content_hash in seen_hashes


def sanitize_text(text: str, max_length: int = 200_000) -> str:
    """Defensively cap text length and strip null bytes before any further
    processing — protects against malformed/malicious PDFs.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    if len(text) > max_length:
        logger.warning("Input text truncated from %d to %d characters", len(text), max_length)
        text = text[:max_length]
    return text
