"""Unit tests for core/chat/response_validator.py."""
from core.chat.response_validator import ResponseValidator


class TestResponseValidator:
    def test_valid_response_with_known_candidate_name(self):
        validator = ResponseValidator()
        result = validator.validate(
            "John Anderson is the strongest candidate for this role.",
            known_candidate_names=["John Anderson", "Sarah Williams"],
        )
        assert result.is_valid is True
        assert result.warnings == []

    def test_flags_unknown_name_like_phrase(self):
        validator = ResponseValidator()
        result = validator.validate(
            "Michael Johnson would be a great fit for this role.",
            known_candidate_names=["John Anderson", "Sarah Williams"],
        )
        assert result.is_valid is False
        assert len(result.warnings) == 1

    def test_no_name_like_phrases_is_valid(self):
        validator = ResponseValidator()
        result = validator.validate(
            "The candidate has strong python skills.",
            known_candidate_names=["John Anderson"],
        )
        assert result.is_valid is True

    def test_empty_response_is_valid(self):
        validator = ResponseValidator()
        result = validator.validate("", known_candidate_names=["John Anderson"])
        assert result.is_valid is True

    def test_partial_name_match_considered_plausibly_known(self):
        validator = ResponseValidator()
        # "John Anderson" mentioned as just "John" wouldn't match this pattern
        # (single word), but a substring match on a known full name should pass.
        result = validator.validate(
            "John Anderson has 8 years of experience.",
            known_candidate_names=["John Anderson"],
        )
        assert result.is_valid is True

    def test_multiple_unknown_names_all_flagged(self):
        validator = ResponseValidator()
        result = validator.validate(
            "Consider Michael Johnson or Emily Clark for this role.",
            known_candidate_names=["John Anderson"],
        )
        assert len(result.warnings) == 2
