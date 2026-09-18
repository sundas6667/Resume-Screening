"""Unit tests for config.py  mainly the ScoringWeights safety rails, since a
silently-broken weight file would corrupt every ranking without an obvious error.
"""
import json

import pytest

from config import COLOR_TOKENS, THEME, ScoringWeights


class TestScoringWeights:
    def test_defaults_sum_to_one(self):
        weights = ScoringWeights()
        weights.validate()  # should not raise

    def test_defaults_match_assignment_spec(self):
        weights = ScoringWeights()
        assert weights.skills == pytest.approx(0.35)
        assert weights.experience == pytest.approx(0.30)
        assert weights.education == pytest.approx(0.15)
        assert weights.certification == pytest.approx(0.10)
        assert weights.project == pytest.approx(0.10)

    def test_rejects_weights_that_do_not_sum_to_one(self):
        bad = ScoringWeights(skills=0.5, experience=0.5, education=0.5, certification=0.1, project=0.1)
        with pytest.raises(ValueError, match="sum to 1.0"):
            bad.validate()

    def test_rejects_out_of_range_weight(self):
        bad = ScoringWeights(skills=1.5, experience=-0.5, education=0.0, certification=0.0, project=0.0)
        with pytest.raises(ValueError):
            bad.validate()

    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "weights.json"
        original = ScoringWeights(skills=0.4, experience=0.3, education=0.1, certification=0.1, project=0.1)
        original.save(path)
        loaded = ScoringWeights.load(path)
        assert loaded == original

    def test_load_falls_back_to_defaults_when_file_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.json"
        loaded = ScoringWeights.load(missing_path)
        assert loaded == ScoringWeights()

    def test_load_falls_back_to_defaults_on_corrupt_json(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json", encoding="utf-8")
        loaded = ScoringWeights.load(path)
        assert loaded == ScoringWeights()

    def test_as_percentages(self):
        weights = ScoringWeights()
        pct = weights.as_percentages()
        assert pct["skills"] == 35
        assert sum(pct.values()) == 100


class TestTheme:
    def test_color_tokens_have_full_scale_for_primary(self):
        assert set(COLOR_TOKENS["primary"].keys()) >= {"50", "500", "600", "900"}

    def test_theme_semantic_aliases_resolve_to_real_colors(self):
        for key in ("background", "surface", "primary", "success", "warning", "danger"):
            assert THEME[key].startswith("#")
