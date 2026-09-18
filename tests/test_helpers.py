"""Unit tests for utils/helpers.py."""
from utils.helpers import clamp, clean_text, compute_content_hash, generate_id, normalize_for_matching, safe_divide, truncate


class TestCleanText:
    """Regression tests: clean_text previously used an ASCII-only allowlist
    that silently deleted every accented/non-Latin character."""

    def test_preserves_accented_latin_characters(self):
        result = clean_text("José García, résumé for François Müller")
        assert "José" in result
        assert "García" in result
        assert "résumé" in result
        assert "Müller" in result

    def test_preserves_arabic_script(self):
        result = clean_text("مرحبا بكم")
        assert result == "مرحبا بكم"

    def test_preserves_cjk_script(self):
        result = clean_text("你好世界")
        assert result == "你好世界"

    def test_strips_control_characters(self):
        result = clean_text("text\x00with\x0bcontrol\x0cchars")
        assert "\x00" not in result
        assert "\x0b" not in result
        assert "\x0c" not in result

    def test_normalizes_windows_line_endings(self):
        result = clean_text("Line1\r\nLine2\r\nLine3")
        assert "\r" not in result
        assert result == "Line1\nLine2\nLine3"

    def test_normalizes_bare_carriage_returns(self):
        result = clean_text("Line1\rLine2")
        assert "\r" not in result

    def test_collapses_excess_blank_lines(self):
        result = clean_text("Line1\n\n\n\n\nLine2")
        assert result == "Line1\n\nLine2"

    def test_strips_bullet_characters(self):
        result = clean_text("\u2022 First item\n\u2022 Second item")
        assert "\u2022" not in result

    def test_empty_input_returns_empty_string(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_collapses_multiple_spaces(self):
        result = clean_text("too    many     spaces")
        assert result == "too many spaces"


class TestNormalizeForMatching:
    def test_lowercases_and_strips_separators(self):
        assert normalize_for_matching("Tensor-Flow") == "tensorflow"
        assert normalize_for_matching("Tensor Flow") == "tensorflow"
        assert normalize_for_matching("TensorFlow") == "tensorflow"

    def test_identical_normalization_for_equivalent_variants(self):
        variants = ["TensorFlow", "tensor-flow", "Tensor_Flow", "tensor.flow"]
        normalized = {normalize_for_matching(v) for v in variants}
        assert len(normalized) == 1


class TestComputeContentHash:
    def test_returns_64_char_hex_digest(self):
        h = compute_content_hash(b"some file bytes")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_identical_bytes_produce_identical_hash(self):
        assert compute_content_hash(b"same content") == compute_content_hash(b"same content")

    def test_different_bytes_produce_different_hash(self):
        assert compute_content_hash(b"content A") != compute_content_hash(b"content B")


class TestGenerateId:
    def test_includes_prefix(self):
        assert generate_id("cand").startswith("cand_")

    def test_ids_are_unique(self):
        ids = {generate_id() for _ in range(500)}
        assert len(ids) == 500


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_zero_denominator_returns_default(self):
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1.0) == -1.0


class TestClamp:
    def test_value_within_range_unchanged(self):
        assert clamp(50) == 50

    def test_value_above_max_clamped(self):
        assert clamp(150) == 100.0

    def test_value_below_min_clamped(self):
        assert clamp(-10) == 0.0


class TestTruncate:
    def test_short_text_unchanged(self):
        assert truncate("short", max_chars=100) == "short"

    def test_long_text_truncated_at_word_boundary(self):
        result = truncate("one two three four five", max_chars=13)
        assert result.endswith("…")
        assert not result[:-1].endswith(" ")
