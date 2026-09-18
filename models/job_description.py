"""
models/job_description.py
==========================
Domain model for the job description a batch of candidates is scored
against. Kept separate from Candidate since it has a different lifecycle
(one per screening session, vs. many candidates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from config import SCHEMA_VERSION


@dataclass
class JobDescription:
    raw_text: str
    title: Optional[str] = None
    cleaned_text: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_years_experience: Optional[float] = None
    required_education_level: Optional[str] = None
    required_certifications: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    schema_version: str = SCHEMA_VERSION

    @property
    def all_relevant_skills(self) -> List[str]:
        """Required + preferred, de-duplicated, preserving order — this is
        the list the skill matcher compares each candidate against.
        """
        seen = set()
        combined = []
        for skill in [*self.required_skills, *self.preferred_skills]:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                combined.append(skill)
        return combined

    @property
    def is_ready(self) -> bool:
        return bool(self.cleaned_text.strip())
