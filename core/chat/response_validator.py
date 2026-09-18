"""
core/chat/response_validator.py
=================================
Lightweight, advisory check that the model's answer doesn't reference a
candidate outside the ones actually provided in context  a defense
against hallucination, not a hard guarantee. Deliberately flags/logs rather
than blocks or rewrites the response: name detection here is a coarse
heuristic (capitalized multi-word phrases), and a false positive would
degrade an otherwise-correct answer worse than a missed true positive would.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from utils.helpers import get_logger

logger = get_logger(__name__)

# Capitalized two-or-more-word sequences look like proper names. Also
# matches plenty of non-names (job titles, tools, cities)  that's why this
# is advisory-only, not a blocking gate.
_NAME_LIKE_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")


@dataclass
class ValidationResult:
    is_valid: bool
    warnings: List[str] = field(default_factory=list)


class ResponseValidator:
    def validate(self, response_text: str, known_candidate_names: List[str]) -> ValidationResult:
        known_normalized: Set[str] = {name.lower() for name in known_candidate_names if name}
        warnings: List[str] = []

        for match in _NAME_LIKE_PATTERN.finditer(response_text or ""):
            phrase = match.group(0)
            if not self._plausibly_known(phrase, known_normalized):
                warnings.append(
                    f"Response mentions '{phrase}', which doesn't match any candidate in the provided context "
                    "(may be a false positive  this check is advisory only)."
                )

        for warning in warnings:
            logger.warning("Response validation: %s", warning)

        return ValidationResult(is_valid=not warnings, warnings=warnings)

    @staticmethod
    def _plausibly_known(phrase: str, known_normalized: Set[str]) -> bool:
        lower = phrase.lower()
        return any(lower in known or known in lower for known in known_normalized)
