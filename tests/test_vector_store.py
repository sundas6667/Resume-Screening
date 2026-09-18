"""Unit tests for core/vector_store.py."""
import pytest

from config import PerformanceConfig
from core.vector_store import VectorStore, VectorStoreError

JD_TEXT = (
    "Senior AI ML Engineer needing Python TensorFlow PyTorch NLP Computer Vision "
    "AWS Docker Kubernetes MLOps experience building production machine learning systems"
)
CANDIDATES = {
    "relevant": (
        "Senior Machine Learning Engineer with Python TensorFlow PyTorch NLP Computer "
        "Vision AWS Docker Kubernetes MLOps deep learning production systems experience"
    ),
    "partial": "Full Stack Developer with React Node.js MongoDB Docker web development",
    "unrelated": "Marketing manager with social media campaigns brand strategy content creation",
}


@pytest.fixture
def built_store() -> VectorStore:
    store = VectorStore()
    store.build(JD_TEXT, CANDIDATES)
    return store


class TestVectorStoreBuild:
    def test_not_built_before_build_called(self):
        assert VectorStore().is_built is False

    def test_is_built_after_build(self, built_store):
        assert built_store.is_built is True

    def test_raises_on_empty_jd(self):
        with pytest.raises(VectorStoreError, match="empty job description"):
            VectorStore().build("", CANDIDATES)

    def test_raises_on_whitespace_only_jd(self):
        with pytest.raises(VectorStoreError):
            VectorStore().build("   \n  ", CANDIDATES)

    def test_raises_on_zero_candidates(self):
        with pytest.raises(VectorStoreError, match="zero candidates"):
            VectorStore().build(JD_TEXT, {})

    def test_raises_on_pure_stopword_documents(self):
        with pytest.raises(VectorStoreError):
            VectorStore().build("the a an of to", {"x": "the a an of to"})

    def test_query_before_build_raises(self):
        with pytest.raises(VectorStoreError, match="build"):
            VectorStore().similarity_scores()


class TestSimilarityScores:
    def test_returns_score_for_every_candidate(self, built_store):
        scores = built_store.similarity_scores()
        assert set(scores.keys()) == set(CANDIDATES.keys())

    def test_scores_are_in_valid_range(self, built_store):
        for score in built_store.similarity_scores().values():
            assert 0.0 <= score <= 100.0

    def test_relevant_candidate_scores_higher_than_unrelated(self, built_store):
        scores = built_store.similarity_scores()
        assert scores["relevant"] > scores["partial"] > scores["unrelated"]

    def test_unrelated_candidate_scores_near_zero(self, built_store):
        assert built_store.similarity_scores()["unrelated"] < 5.0

    def test_scores_are_cached_between_calls(self, built_store):
        first = built_store.similarity_scores()
        second = built_store.similarity_scores()
        assert first is second  # same dict object -> confirms caching, not recomputation

    def test_get_similarity_for_known_candidate(self, built_store):
        assert built_store.get_similarity("relevant") == built_store.similarity_scores()["relevant"]

    def test_get_similarity_for_unknown_candidate_returns_none(self, built_store):
        assert built_store.get_similarity("nonexistent") is None


class TestSimilaritySearch:
    def test_reuses_fitted_vocabulary_without_rebuilding(self, built_store):
        results = built_store.similarity_search("cloud infrastructure Docker Kubernetes AWS")
        assert results[0][0] == "relevant"

    def test_respects_top_k(self, built_store):
        results = built_store.similarity_search("machine learning", top_k=2)
        assert len(results) == 2

    def test_empty_query_returns_empty_list(self, built_store):
        assert built_store.similarity_search("") == []

    def test_results_sorted_descending(self, built_store):
        results = built_store.similarity_search("Python TensorFlow AWS", top_k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestVocabularySize:
    def test_returns_positive_vocabulary_size(self, built_store):
        assert built_store.vocabulary_size() > 0

    def test_respects_max_features_config(self):
        store = VectorStore(performance_config=PerformanceConfig(tfidf_max_features=5))
        store.build(JD_TEXT, CANDIDATES)
        assert store.vocabulary_size() <= 5


class TestVectorAccessors:
    def test_get_candidate_vector_returns_row_for_known_candidate(self, built_store):
        vector = built_store.get_candidate_vector("relevant")
        assert vector is not None
        assert vector.shape[0] == 1

    def test_get_candidate_vector_returns_none_for_unknown_candidate(self, built_store):
        assert built_store.get_candidate_vector("nonexistent") is None

    def test_get_job_vector_returns_single_row(self, built_store):
        vector = built_store.get_job_vector()
        assert vector.shape[0] == 1

    def test_candidate_and_job_vectors_have_same_dimensionality(self, built_store):
        assert built_store.get_candidate_vector("relevant").shape[1] == built_store.get_job_vector().shape[1]


class TestClearCache:
    def test_clear_cache_forces_recomputation(self, built_store):
        first = built_store.similarity_scores()
        built_store.clear_cache()
        second = built_store.similarity_scores()
        assert first == second
        assert first is not second  # different dict objects -> was actually recomputed

    def test_store_still_queryable_after_clear_cache(self, built_store):
        built_store.clear_cache()
        assert built_store.get_similarity("relevant") is not None


class TestMetadata:
    def test_metadata_before_build(self):
        meta = VectorStore().metadata
        assert meta["is_built"] is False
        assert meta["candidate_count"] == 0

    def test_metadata_after_build(self, built_store):
        meta = built_store.metadata
        assert meta["is_built"] is True
        assert meta["candidate_count"] == 3
        assert meta["vocabulary_size"] > 0
        assert meta["built_at"] is not None
