"""Unit tests for core/parser.py  uses reportlab to generate real PDF bytes
in-memory so these tests exercise the actual PyPDF2 code path, not a mock.
"""
import io

import pytest
from PyPDF2 import PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from config import PerformanceConfig
from core.parser import CorruptedPdfError, EncryptedPdfError, ParsingError, ResumeParser, ScannedPdfError


def make_pdf_bytes(lines: list[str], pages: int = 1) -> bytes:
    """Build a minimal real PDF with the given lines of text, for testing."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for _ in range(pages):
        y = 750
        for line in lines:
            c.drawString(72, y, line)
            y -= 20
        c.showPage()
    c.save()
    return buf.getvalue()


def make_encrypted_pdf_bytes(password: str = "secret123") -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def parser() -> ResumeParser:
    return ResumeParser()


class TestParseValidPdf:
    def test_extracts_text_correctly(self, parser):
        pdf = make_pdf_bytes([
            "Jane Doe", "jane@example.com | +1 555-123-4567",
            "SKILLS", "Python, AWS, Docker, Kubernetes, PostgreSQL, FastAPI",
        ])
        doc = parser.parse(pdf, "jane.pdf")
        assert "Jane Doe" in doc.cleaned_text
        assert "Python, AWS, Docker" in doc.cleaned_text

    def test_records_correct_page_count(self, parser):
        pdf = make_pdf_bytes(["Software Engineer with five years of professional experience in backend systems."], pages=2)
        doc = parser.parse(pdf, "multi.pdf")
        assert doc.page_count == 2

    def test_records_filename_and_size(self, parser):
        pdf = make_pdf_bytes(["Experienced backend engineer specializing in distributed systems and cloud infrastructure."])
        doc = parser.parse(pdf, "resume.pdf")
        assert doc.filename == "resume.pdf"
        assert doc.file_size_bytes == len(pdf)

    def test_computes_content_hash(self, parser):
        pdf = make_pdf_bytes(["Full stack developer with expertise in React, Node.js, and PostgreSQL databases."])
        doc = parser.parse(pdf, "a.pdf")
        assert len(doc.content_hash) == 64  # sha256 hex digest length

    def test_identical_bytes_produce_identical_hash(self, parser):
        pdf = make_pdf_bytes(["Data scientist with a background in machine learning, statistics, and Python programming."])
        doc1 = parser.parse(pdf, "a.pdf")
        doc2 = parser.parse(pdf, "b.pdf")  # different filename, same bytes
        assert doc1.content_hash == doc2.content_hash

    def test_records_positive_parse_time(self, parser):
        pdf = make_pdf_bytes(["DevOps engineer experienced with Docker, Kubernetes, Terraform, and CI/CD pipelines."])
        doc = parser.parse(pdf, "a.pdf")
        assert doc.parse_time_ms >= 0.0


class TestParseInvalidPdf:
    def test_rejects_encrypted_pdf(self, parser):
        pdf = make_encrypted_pdf_bytes()
        with pytest.raises(ParsingError, match="password-protected"):
            parser.parse(pdf, "locked.pdf")

    def test_rejects_corrupted_bytes(self, parser):
        with pytest.raises(ParsingError):
            parser.parse(b"this is definitely not a pdf", "bad.pdf")

    def test_rejects_empty_bytes(self, parser):
        with pytest.raises(ParsingError):
            parser.parse(b"", "empty.pdf")

    def test_rejects_pdf_with_no_extractable_text(self, parser):
        # A blank page has zero text content -> should be flagged as likely scanned
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)
        with pytest.raises(ParsingError, match="scanned"):
            parser.parse(buf.getvalue(), "blank.pdf")


class TestParsingErrorSubclasses:
    """Regression tests: specific exception types let future callers (e.g. an
    OCR fallback) react to each failure reason without string-matching."""

    def test_encrypted_raises_encrypted_subclass(self, parser):
        with pytest.raises(EncryptedPdfError):
            parser.parse(make_encrypted_pdf_bytes(), "locked.pdf")

    def test_corrupted_bytes_raise_corrupted_subclass(self, parser):
        with pytest.raises(CorruptedPdfError):
            parser.parse(b"not a pdf at all", "bad.pdf")

    def test_scanned_pdf_raises_scanned_subclass(self, parser):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)
        with pytest.raises(ScannedPdfError):
            parser.parse(buf.getvalue(), "blank.pdf")

    def test_all_subclasses_are_catchable_as_base_parsing_error(self, parser):
        for pdf_bytes, filename in [
            (make_encrypted_pdf_bytes(), "locked.pdf"),
            (b"garbage", "bad.pdf"),
        ]:
            with pytest.raises(ParsingError):
                parser.parse(pdf_bytes, filename)


class TestParseBatch:
    def test_isolates_one_bad_file_from_the_rest(self, parser):
        good = make_pdf_bytes(["Mobile developer with experience shipping iOS and Android apps used by millions."])
        successes, failures = parser.parse_batch([
            (good, "good.pdf"),
            (b"corrupt", "bad.pdf"),
        ])
        assert len(successes) == 1
        assert successes[0].filename == "good.pdf"
        assert len(failures) == 1
        assert failures[0][0] == "bad.pdf"

    def test_deduplicates_identical_content(self, parser):
        good = make_pdf_bytes(["QA engineer with strong background in test automation using Selenium and Pytest frameworks."])
        successes, failures = parser.parse_batch([
            (good, "first.pdf"),
            (good, "second_copy.pdf"),
        ])
        assert len(successes) == 1
        assert any("Duplicate" in msg for _, msg in failures)

    def test_respects_batch_size_limit(self):
        parser = ResumeParser(performance_config=PerformanceConfig(max_uploads_per_batch=2))
        files = [
            (make_pdf_bytes([f"Candidate number {i} with several years of relevant professional experience."]), f"r{i}.pdf")
            for i in range(4)
        ]
        successes, failures = parser.parse_batch(files)
        assert len(successes) == 2
        assert len(failures) == 2
        assert all("Batch limit" in msg for _, msg in failures)

    def test_empty_batch_returns_empty_results(self, parser):
        successes, failures = parser.parse_batch([])
        assert successes == []
        assert failures == []
