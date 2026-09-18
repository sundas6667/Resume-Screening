"""
utils/helpers.py
=================
Small, dependency free helper functions reused across the parsing, ranking,
and UI layers: logging setup, text normalization, timing instrumentation,
and ID generation. Nothing here should know about Streamlit, PDFs, or
scoring — those live in their own modules.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import re
import time
import uuid
from logging.handlers import RotatingFileHandler
from typing import Callable, TypeVar

from config import LOGS_DIR
from utils.constants import (
    BULLET_CHAR_PATTERN,
    CONTROL_CHAR_PATTERN,
    MULTI_NEWLINE_PATTERN,
    MULTI_SPACE_PATTERN,
)

F = TypeVar("F", bound=Callable)

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "resume_screening") -> logging.Logger:
    """Return a configured logger, writing to logs/app.log and stdout.

    Cached per-name so repeated calls (e.g. one per module) don't stack
    duplicate handlers.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # All INFO+ messages (resume uploaded, parsed, ranked, etc.)
        app_handler = RotatingFileHandler(
            LOGS_DIR / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        app_handler.setFormatter(formatter)
        logger.addHandler(app_handler)

        # WARNING+ only, split out so failures are visible without grepping
        # through routine activity logs — cheap to add, valuable when
        # triaging a bug report after the fact.
        error_handler = RotatingFileHandler(
            LOGS_DIR / "error.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    _LOGGERS[name] = logger
    return logger


def timed(logger: logging.Logger | None = None) -> Callable[[F], F]:
    """Decorator that logs how long a function took — used on the hot path
    (parsing, vectorizing, ranking) to keep an eye on the <10s/10-resume budget.
    """
    log = logger or get_logger()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info("%s completed in %.1fms", func.__qualname__, elapsed_ms)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator


def generate_id(prefix: str = "cand") -> str:
    """Short, collision-resistant ID for a candidate/session object."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def clean_text(raw_text: str) -> str:
    """Normalize whitespace/bullets and strip non-printable characters from
    text extracted out of a PDF, without destroying line structure (section
    detection depends on line breaks surviving this step).
    """
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_PATTERN.sub("", text)
    text = BULLET_CHAR_PATTERN.sub("", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """Truncate text for display in tables/cards without cutting mid-word."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + suffix


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises on a zero denominator — used throughout
    the scoring engine where a candidate may have zero listed skills, etc.
    """
    return numerator / denominator if denominator else default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a score into a valid display range."""
    return max(low, min(high, value))


def compute_content_hash(data: bytes) -> str:
    """SHA-256 of raw file bytes — used to detect a resume uploaded twice
    (accidentally, or across two separate batches) so it isn't double-counted
    in rankings or analytics.
    """
    return hashlib.sha256(data).hexdigest()


_NORMALIZE_PATTERN = re.compile(r"[\s\-_./]+")


def normalize_for_matching(text: str) -> str:
    """Fold a skill/token to a comparison-safe form: lowercase, and spaces/
    hyphens/underscores/dots collapsed away. This lets the skill matcher
    treat "Tensor Flow", "tensor-flow", and "TensorFlow" as the same token
    without needing an explicit alias for every spacing variant — aliases
    in skills.json are reserved for genuinely different tokens (e.g. "JS").
    """
    return _NORMALIZE_PATTERN.sub("", text.strip().lower())
