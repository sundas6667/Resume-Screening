"""
core/chat/prompt_builder.py
============================
Builds the system prompt and per-question message list sent to Groq.

This is the module most directly responsible for the "never hallucinate,
use only extracted candidate information" requirement: the ONLY candidate
data that reaches the model is what's serialized here from
Candidate.to_summary_dict() (which already excludes raw_text) — nothing
else, and nothing invented. The context block is rebuilt fresh for every
question (reflecting whichever candidates the retriever just selected)
rather than accumulated in history, so it never grows unbounded and always
matches what's actually being asked about.
"""
from __future__ import annotations

import json
from typing import Dict, List

from config import PROMPT_VERSION
from core.chat.chat_history import ChatHistory
from models.candidate import Candidate
from models.job_description import JobDescription

_SYSTEM_PROMPT = """You are a professional HR assistant helping a recruiter evaluate job candidates for a specific role.

Rules you must always follow:
1. Only use the candidate data provided in the CANDIDATE DATA block below. Never invent, assume, or infer any fact about a candidate that isn't explicitly present in that data.
2. If asked about a candidate or a detail not present in the provided data, say so clearly instead of guessing.
3. When comparing candidates or explaining a ranking, cite the specific scores, skills, or experience that justify your answer.
4. Be concise and professional — you are briefing a busy recruiter, not writing an essay.
5. If a candidate's parser_confidence is low or parsing_warnings are listed, mention that their data may be incomplete when it's relevant to your answer.
6. Scores and rankings are already computed by the system — never recalculate, re-estimate, or override a score yourself; only reference the ones provided.
"""


class PromptBuilder:
    """Stateless — safe to share a single instance across a whole session."""

    version: str = PROMPT_VERSION

    def build_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_context_block(self, candidates: List[Candidate], job_description: JobDescription) -> str:
        """Serialize exactly the structured facts the model is allowed to
        use. Candidate.to_summary_dict() is the single source of truth for
        "what the chatbot is allowed to know" — extending what the
        chatbot can discuss means extending that method, not this one.
        """
        jd_block = {
            "title": job_description.title,
            "required_skills": job_description.required_skills,
            "preferred_skills": job_description.preferred_skills,
            "min_years_experience": job_description.min_years_experience,
            "required_education_level": job_description.required_education_level,
        }
        payload = {
            "job_description": jd_block,
            "candidates": [c.to_summary_dict() for c in candidates],
        }
        return "CANDIDATE DATA (JSON):\n" + json.dumps(payload, indent=2, default=str)

    def build_messages(
        self,
        question: str,
        candidates: List[Candidate],
        job_description: JobDescription,
        history: ChatHistory,
    ) -> List[Dict[str, str]]:
        """Assemble the full message list: persona + fresh data context +
        conversational history (text only, no embedded data) + new question.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "system", "content": self.build_context_block(candidates, job_description)},
        ]
        messages.extend(history.to_groq_format())
        messages.append({"role": "user", "content": question})
        return messages
