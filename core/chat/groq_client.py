"""
core/chat/groq_client.py
=========================
Thin wrapper around the Groq SDK  the only module in the chat subsystem
that talks to the network. Knows nothing about candidates, prompts, or
ranking; just sends messages and returns text, with specific, graceful
error handling per failure type (never crashes the app).
"""
from __future__ import annotations

import time
from typing import Dict, List

from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from config import GROQ_MAX_TOKENS, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_TIMEOUT_SECONDS
from utils.helpers import get_logger

logger = get_logger(__name__)


class ChatError(Exception):
    """Raised when the Groq API call fails for any reason  always caught
    by the caller (core.chat.chatbot.HRChatbot) and turned into a friendly,
    user-facing message. Never allowed to propagate and crash the app.
    """


class GroqClient:
    def __init__(self, api_key: str, model: str = GROQ_MODEL):
        if not api_key or not api_key.strip():
            raise ChatError("No Groq API key provided , enter one in the sidebar to use the AI assistant.")
        self.model = model
        self._client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS)

    def send(
        self,
        messages: List[Dict[str, str]],
        temperature: float = GROQ_TEMPERATURE,
        max_tokens: int = GROQ_MAX_TOKENS,
    ) -> str:
        """Send a message list to Groq and return the assistant's reply text.

        Raises:
            ChatError: on any failure (auth, rate limit, timeout, connection,
                or an unexpected SDK error) — always with a message safe to
                show the user directly.
        """
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
        except AuthenticationError as exc:
            raise ChatError("Invalid Groq API key — please check the key entered in the sidebar.") from exc
        except RateLimitError as exc:
            raise ChatError("Groq API rate limit reached — please wait a moment and try again.") from exc
        except APITimeoutError as exc:
            raise ChatError("The request to Groq timed out — please try again.") from exc
        except APIConnectionError as exc:
            raise ChatError("Could not connect to Groq — please check your internet connection.") from exc
        except APIError as exc:
            raise ChatError(f"Groq API error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - never crash on an unexpected SDK error
            logger.error("Unexpected error calling Groq: %s", exc)
            raise ChatError("An unexpected error occurred while contacting the AI assistant.") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Groq request completed in %.0fms", elapsed_ms)

        if not response.choices:
            raise ChatError("Groq returned an empty response — please try rephrasing your question.")
        return response.choices[0].message.content or ""
