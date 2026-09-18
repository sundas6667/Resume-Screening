"""Unit tests for core/chat/retriever.py."""
from core.chat.retriever import CandidateRetriever
from core.vector_store import VectorStore
from models.candidate import Candidate, RankingResult


def make_candidate(name: str, score: float = 0.0) -> Candidate:
    c = Candidate(full_name=name)
    c.ranking = RankingResult(overall_score=score)
    return c


class TestSmallBatchBehavior:
    def test_small_batch_returns_all_candidates(self):
        candidates = [make_candidate(f"Candidate {i}") for i in range(5)]
        retriever = CandidateRetriever()
        result = retriever.retrieve("who is best?", candidates)
        assert result == candidates

    def test_empty_candidate_list(self):
        retriever = CandidateRetriever()
        assert retriever.retrieve("anything", []) == []


class TestNameMatching:
    def test_question_naming_one_candidate_returns_just_that_one(self):
        candidates = [make_candidate(f"Person {i} Anderson") for i in range(15)]
        candidates.append(make_candidate("John Smith"))
        retriever = CandidateRetriever()
        result = retriever.retrieve("Tell me about John Smith's experience", candidates)
        assert len(result) == 1
        assert result[0].full_name == "John Smith"

    def test_question_naming_two_candidates_returns_both(self):
        candidates = [make_candidate(f"Person {i} Anderson") for i in range(15)]
        candidates.append(make_candidate("John Smith"))
        candidates.append(make_candidate("Sarah Williams"))
        retriever = CandidateRetriever()
        result = retriever.retrieve("Compare John Smith and Sarah Williams", candidates)
        names = {c.full_name for c in result}
        assert names == {"John Smith", "Sarah Williams"}

    def test_common_first_name_alone_does_not_falsely_match_everyone(self):
        candidates = [make_candidate(f"Sam Person{i}") for i in range(15)]
        retriever = CandidateRetriever()
        # A question mentioning "sample" should not match every "Sam ___" candidate
        result = retriever.retrieve("Show me a sample of top candidates", candidates)
        # Falls through to top-ranked fallback, not an accidental name match on every "Sam"
        assert len(result) <= 8


class TestVectorStoreFallback:
    def test_uses_vector_store_when_no_name_match(self):
        candidates = [make_candidate(f"Candidate {i}", score=i) for i in range(15)]
        vector_store = VectorStore()
        vector_store.build(
            "Need Python cloud infrastructure experience",
            {c.candidate_id: f"Python cloud experience level {i}" for i, c in enumerate(candidates)},
        )
        retriever = CandidateRetriever(vector_store=vector_store)
        result = retriever.retrieve("who has cloud infrastructure experience", candidates, top_k=5)
        assert len(result) <= 5

    def test_falls_back_to_top_ranked_when_no_vector_store(self):
        candidates = [make_candidate(f"Candidate {i}", score=i) for i in range(15)]
        retriever = CandidateRetriever(vector_store=None)
        result = retriever.retrieve("general question with no name", candidates, top_k=5)
        assert len(result) == 5
        assert result[0].ranking.overall_score == 14  # highest score first

    def test_falls_back_to_top_ranked_when_vector_store_not_built(self):
        candidates = [make_candidate(f"Candidate {i}", score=i) for i in range(15)]
        retriever = CandidateRetriever(vector_store=VectorStore())  # never built
        result = retriever.retrieve("general question", candidates, top_k=3)
        assert len(result) == 3
