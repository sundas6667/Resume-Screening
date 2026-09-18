"""
core/parser.py
===============
Turns raw PDF bytes into clean, structured-ready text. This module owns
exactly one responsibility: bytes in, text out. It knows nothing about
skills, ranking, or Streamlit  core/extractor.py is what turns this text
into a Candidate.

Assumes basic file-level checks (extension, size) already happened at the
upload boundary via utils.validators.validate_uploaded_file , this module
handles the checks that can only be done by actually attempting to open the
PDF (encrypted / corrupted / scanned-with-no-text).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PyPDF2 import PdfReader
from PyPDF2.errors import PyPdfError

from config import PerformanceConfig
from utils.helpers import clean_text, compute_content_hash, get_logger, timed
from utils.validators import validate_pdf_content

logger = get_logger(__name__)


class ParsingError(Exception):
    """Raised when a PDF cannot be turned into usable text  encrypted,
    corrupted, empty, or scanned with no extractable text layer.

    Always caught by the calling pipeline (never allowed to crash the app);
    the message is written to be shown directly to the user, not just logged.
    Callers that only care "did parsing fail" can catch this base class;
    callers that need to react differently per failure reason (e.g. a future
    OCR fallback specifically for scanned PDFs) can catch the subclasses below.
    """


class EncryptedPdfError(ParsingError):
    """The PDF is password-protected and could not be read without a password."""


class CorruptedPdfError(ParsingError):
    """The PDF bytes could not be opened/parsed at all, or contain zero pages."""


class ScannedPdfError(ParsingError):
    """The PDF opened fine but yielded ~no extractable text  almost always
    means it's a scanned/image-based document with no text layer. Kept as
    its own type (not just an error message) so a future OCR pipeline can
    catch this specific condition programmatically:

        try:
            doc = parser.parse(file_bytes, filename)
        except ScannedPdfError:
            doc = ocr_pipeline.process(file_bytes, filename)  # not yet implemented
    """


@dataclass
class ParsedDocument:
    """Output of ResumeParser.parse()  plain extracted text plus enough
    provenance to populate a Candidate's ResumeMetadata downstream.
    """
    filename: str
    raw_text: str
    cleaned_text: str
    page_count: int
    file_size_bytes: int
    content_hash: str
    parse_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


class ResumeParser:
    """Extracts and cleans text from PDF resumes.

    Takes an optional PerformanceConfig via constructor injection rather
    than importing the global config directly, so it can be unit tested
    (or reconfigured) without monkeypatching a module-level import.
    """

    def __init__(self, performance_config: Optional[PerformanceConfig] = None):
        self.performance_config = performance_config or PerformanceConfig()

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse a single PDF's bytes into a ParsedDocument.

        Raises:
            ParsingError: if the PDF is encrypted, empty, corrupted, or has
                no extractable text (likely scanned). Callers  see
                `parse_batch`  are expected to catch this per file and
                continue processing the rest of the batch.
        """
        start = time.perf_counter()
        content_hash = compute_content_hash(file_bytes)

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except PyPdfError as exc:
            logger.warning("Failed to open '%s': %s", filename, exc)
            raise CorruptedPdfError(
                f"'{filename}' could not be read  the file may be corrupted or not a valid PDF."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - last-resort guard, never crash on a bad upload
            logger.error("Unexpected error opening '%s': %s", filename, exc)
            raise CorruptedPdfError(f"'{filename}' could not be opened due to an unexpected error.") from exc

        is_encrypted = reader.is_encrypted
        # Accessing .pages on an encrypted, non-decrypted reader raises
        # FileNotDecryptedError, so we must check is_encrypted first.
        page_count = 0 if is_encrypted else len(reader.pages)

        raw_text = "" if is_encrypted else self._extract_text(reader, filename)

        validation = validate_pdf_content(
            is_encrypted=is_encrypted,
            page_count=page_count,
            extracted_char_count=len(raw_text.strip()),
        )
        if not validation.is_valid:
            logger.warning("Validation failed for '%s': %s", filename, validation.message)
            if is_encrypted:
                raise EncryptedPdfError(validation.message)
            if page_count == 0:
                raise CorruptedPdfError(validation.message)
            raise ScannedPdfError(validation.message)

        cleaned = clean_text(raw_text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Parsed '%s' in %.1fms (%d pages, %d chars)", filename, elapsed_ms, page_count, len(cleaned)
        )

        return ParsedDocument(
            filename=filename,
            raw_text=raw_text,
            cleaned_text=cleaned,
            page_count=page_count,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            parse_time_ms=elapsed_ms,
        )

    @staticmethod
    def _extract_text(reader: PdfReader, filename: str) -> str:
        """Extract text page by page  tolerates a single bad page (rare
        but possible with malformed PDFs) without failing the whole document.
        """
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                pages_text.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - PyPDF2 can raise assorted internal errors per-page
                logger.warning("Could not extract text from page %d of '%s': %s", i + 1, filename, exc)
                pages_text.append("")
        return "\n".join(pages_text)

    @timed(logger)
    def parse_batch(
        self, files: List[Tuple[bytes, str]]
    ) -> Tuple[List[ParsedDocument], List[Tuple[str, str]]]:
        """Parse multiple files, isolating failures so one bad PDF doesn't
        abort the batch.

        Returns:
            (successful_documents, [(filename, error_message), ...])
        """
        limit = self.performance_config.max_uploads_per_batch
        overflow: List[Tuple[bytes, str]] = []
        if len(files) > limit:
            logger.warning("Batch of %d exceeds limit of %d; extra files will be skipped", len(files), limit)
            files, overflow = files[:limit], files[limit:]

        successes: List[ParsedDocument] = []
        failures: List[Tuple[str, str]] = []
        seen_hashes: set[str] = set()

        for file_bytes, filename in files:
            try:
                doc = self.parse(file_bytes, filename)
            except ParsingError as exc:
                failures.append((filename, str(exc)))
                continue

            if doc.content_hash in seen_hashes:
                logger.info("Skipping '%s' — duplicate content of an already-processed file", filename)
                failures.append((filename, "Duplicate of an already uploaded resume  skipped."))
                continue

            seen_hashes.add(doc.content_hash)
            successes.append(doc)

        for _, filename in overflow:
            failures.append(
                (filename, f"Batch limit of {limit} resumes exceeded  this file was not processed.")
            )

        return successes, failures
