"""
core/chat/chat_history.py
==========================
In-memory conversation history for the HR chatbot. Session-scoped  backed
by whatever the caller holds (Streamlit session_state in the Module 7 UI),
not persisted to disk. Deliberately separate from prompt_builder.py: this
module only knows about ordered (role, content) turns, nothing about
candidates or job descriptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ChatHistory:
    """Ordered list of ChatMessages plus the bookkeeping needed to keep a
    growing conversation from blowing past the model's context window 
    only the `max_messages` most recent turns are ever sent to Groq, though
    the full history is retained for display in the UI.
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: List[ChatMessage] = []

    def add_user_message(self, content: str) -> None:
        self._messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(ChatMessage(role="assistant", content=content))

    @property
    def messages(self) -> List[ChatMessage]:
        """Full history, for UI display  use to_groq_format() for what
        actually gets sent to the model (which is bounded/truncated).
        """
        return list(self._messages)

    def recent(self, n: Optional[int] = None) -> List[ChatMessage]:
        limit = n if n is not None else self.max_messages
        return self._messages[-limit:] if limit > 0 else []

    def to_groq_format(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Convert recent history to the {"role": ..., "content": ...}
        shape the Groq chat completions API expects.
        """
        return [{"role": m.role, "content": m.content} for m in self.recent(n)]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
