"""
core/chat/chatbot.py
=====================
Orchestrates the pipeline: already computed candidate/ranking data ->
Retriever -> Prompt Builder -> Groq API -> Answer.

This module deliberately does NOT recompute anything Modules 1-4 already
produced , no re-parsing resumes, no rerunning TF-IDF, no recalculating
scores. It only reads Candidate/MatchingResult/RankingResult as already-
computed facts, which is what keeps the chatbot's answers consistent with
whatever the recruiter sees elsewhere in the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config import PROMPT_VERSION, SIMILARITY_ENGINE_VERSION
from core.chat.chat_history import ChatHistory
from core.chat.groq_client import ChatError, GroqClient
from core.chat.prompt_builder import PromptBuilder
from core.chat.response_validator import ResponseValidator
from core.chat.retriever import CandidateRetriever
from core.vector_store import VectorStore
from models.candidate import Candidate
from models.job_description import JobDescription
from utils.helpers import generate_id, get_logger

logger = get_logger(__name__)

SUGGESTED_QUESTIONS = [
    "Who is the best candidate for this role?",
    "Compare the top 2 candidates.",
    "Which candidates are missing the most required skills?",
    "Who has the strongest leadership experience?",
    "Which candidate has the most relevant cloud experience?",
    "Recommend interview questions for the top candidate.",
]


@dataclass
class RetrievalDiagnostics:
    """Metadata about what informed the most recent answer  covers source
    attribution ("based on John Anderson, Sarah Williams"), retrieval
    quality (similarity scores), and versioning, in one place rather than
    three overlapping trackers. Mainly for logs/debugging; a UI MAY choose
    to surface a summary of this (e.g. "Based on 2 candidates' data").
    """
    retrieved_candidate_names: List[str] = field(default_factory=list)
    retrieved_count: int = 0
    top_similarity: Optional[float] = None
    average_similarity: Optional[float] = None
    prompt_version: str = PROMPT_VERSION
    vector_store_version: str = SIMILARITY_ENGINE_VERSION
    question_number: int = 0
    answered_at: datetime = field(default_factory=datetime.now)


class HRChatbot:
    """One instance per screening session (one JD + one candidate batch).
    Accepts its collaborators via constructor injection (DI)  a UI layer
    (Module 7) is expected to construct one of these per session and reuse
    it across turns so ChatHistory persists.
    """

    def __init__(
        self,
        api_key: str,
        candidates: List[Candidate],
        job_description: JobDescription,
        vector_store: Optional[VectorStore] = None,
        history: Optional[ChatHistory] = None,
        client: Optional[GroqClient] = None,
    ):
        self.client = client or GroqClient(api_key=api_key)
        self.candidates = candidates
        self.job_description = job_description
        self.retriever = CandidateRetriever(vector_store=vector_store)
        self.prompt_builder = PromptBuilder()
        self.validator = ResponseValidator()
        self.history = history or ChatHistory()
        self.session_id = generate_id("chatsession")
        self.last_retrieval: Optional[RetrievalDiagnostics] = None

    def ask(self, question: str) -> str:
        """Answer one question. Never raises for expected failure modes —
        returns a friendly, user-facing string instead.
        """
        if not question or not question.strip():
            return "Please ask a question about the candidates or the job description."

        try:
            relevant_candidates = self.retriever.retrieve(question, self.candidates)
            messages = self.prompt_builder.build_messages(
                question, relevant_candidates, self.job_description, self.history
            )
            answer = self.client.send(messages)
        except ChatError as exc:
            logger.warning("Chat request failed: %s", exc)
            return f"⚠️ {exc}"

        self.validator.validate(answer, [c.display_name for c in self.candidates])
        # Validation warnings are logged (see ResponseValidator) rather than
        # altering the answer — see the module docstring for why this stays
        # advisory rather than a blocking gate.

        similarities = [c.matching.similarity_score for c in relevant_candidates if c.matching]
        self.last_retrieval = RetrievalDiagnostics(
            retrieved_candidate_names=[c.display_name for c in relevant_candidates],
            retrieved_count=len(relevant_candidates),
            top_similarity=round(max(similarities), 1) if similarities else None,
            average_similarity=round(sum(similarities) / len(similarities), 1) if similarities else None,
            question_number=len(self.history) // 2 + 1,
        )
        logger.info(
            "Session %s, question #%d: retrieved %d candidate(s), avg similarity %s",
            self.session_id, self.last_retrieval.question_number,
            self.last_retrieval.retrieved_count, self.last_retrieval.average_similarity,
        )

        self.history.add_user_message(question)
        self.history.add_assistant_message(answer)
        return answer

    def clear_history(self) -> None:
        self.history.clear()

    @staticmethod
    def suggested_questions() -> List[str]:
        return list(SUGGESTED_QUESTIONS)
