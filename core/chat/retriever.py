"""
core/chat/retriever.py
=======================
Decides which candidates' data is relevant to a chatbot question, so the
prompt builder isn't forced to include every candidate's full summary for
every question  wasteful for large batches, and dilutes the model's focus
for targeted questions like "who has cloud experience".

Reuses the VectorStore already built during ranking (core/matcher.py) via
similarity_search() rather than rebuilding it  this is the concrete
payoff of Module 3's "reuse vectors during chatbot queries" requirement.
"""
from __future__ import annotations

import re
from typing import List, Optional

from core.vector_store import VectorStore, VectorStoreError
from models.candidate import Candidate
from utils.helpers import get_logger

logger = get_logger(__name__)

# Below this many candidates, just include everyone  the context-size cost
# of including all of them is smaller than the risk of a heuristic wrongly
# excluding someone relevant to a general question.
_SMALL_BATCH_THRESHOLD = 12
_DEFAULT_TOP_K = 8


class CandidateRetriever:
    """Selects which candidates to surface in a chatbot prompt for a given
    question. Stateless aside from the (optional, reused) vector store.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store

    def retrieve(self, question: str, candidates: List[Candidate], top_k: int = _DEFAULT_TOP_K) -> List[Candidate]:
        """Return the subset of `candidates` relevant to `question`."""
        if not candidates:
            return []
        if len(candidates) <= _SMALL_BATCH_THRESHOLD:
            return candidates

        named = self._match_by_name(question, candidates)
        if named:
            return named

        if self.vector_store is not None and self.vector_store.is_built:
            try:
                ranked = self.vector_store.similarity_search(question, top_k=top_k)
                matched_ids = {candidate_id for candidate_id, _ in ranked}
                matched = [c for c in candidates if c.candidate_id in matched_ids]
                if matched:
                    return matched
            except VectorStoreError as exc:
                logger.warning("Retriever similarity search failed, falling back to top-ranked: %s", exc)

        # Fallback: top-ranked candidates by whatever ranking already exists.
        return sorted(candidates, key=lambda c: c.ranking.overall_score, reverse=True)[:top_k]

    @staticmethod
    def _match_by_name(question: str, candidates: List[Candidate]) -> List[Candidate]:
        """If the question names one or more specific candidates (e.g.
        "compare John and Sarah"), return just those  a comparison
        question wants exactly the named candidates, not the whole batch.
        """
        lower_question = question.lower()
        matched = []
        for candidate in candidates:
            name = candidate.display_name
            if not name or name == "Unknown Candidate":
                continue
            name_parts = name.split()
            # Matching on the last name (when available) avoids a common
            # first name like "Sam" over-matching unrelated question text.
            check_token = name_parts[-1] if len(name_parts) >= 2 else name
            if re.search(rf"\b{re.escape(check_token.lower())}\b", lower_question):
                matched.append(candidate)
        return matched
