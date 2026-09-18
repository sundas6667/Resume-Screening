"""core.chat , the AI HR Assistant subsystem (Groq-powered).

Structure (each file owns exactly one responsibility):
    groq_client.py       - network calls to Groq, nothing else
    prompt_builder.py     - system prompt + per-question context/messages
    chat_history.py       - conversation turn storage
    retriever.py          - which candidates are relevant to this question
    response_validator.py - advisory hallucination check
    chatbot.py             - orchestrates the above into one .ask() call
"""
from core.chat.chat_history import ChatHistory, ChatMessage
from core.chat.chatbot import HRChatbot, RetrievalDiagnostics
from core.chat.groq_client import ChatError, GroqClient
from core.chat.prompt_builder import PromptBuilder
from core.chat.response_validator import ResponseValidator
from core.chat.retriever import CandidateRetriever

__all__ = [
    "HRChatbot",
    "RetrievalDiagnostics",
    "ChatHistory",
    "ChatMessage",
    "ChatError",
    "GroqClient",
    "PromptBuilder",
    "ResponseValidator",
    "CandidateRetriever",
]
