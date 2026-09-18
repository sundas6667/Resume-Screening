"""Unit tests for core/chat/groq_client.py. Mocks the Groq SDK client so
these tests never make a real network call.
"""
from unittest.mock import MagicMock, patch

import pytest
from groq import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from core.chat.groq_client import ChatError, GroqClient


def make_fake_response(content: str = "This is the AI's answer.") -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestGroqClientConstruction:
    def test_raises_without_api_key(self):
        with pytest.raises(ChatError, match="No Groq API key"):
            GroqClient(api_key="")

    def test_raises_with_whitespace_only_key(self):
        with pytest.raises(ChatError):
            GroqClient(api_key="   ")

    @patch("core.chat.groq_client.Groq")
    def test_constructs_with_valid_key(self, mock_groq_cls):
        client = GroqClient(api_key="fake-key-123")
        mock_groq_cls.assert_called_once()
        assert client.model  # uses the configured default model


class TestGroqClientSend:
    @patch("core.chat.groq_client.Groq")
    def test_returns_response_content(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.return_value = make_fake_response("Hello from the model.")
        client = GroqClient(api_key="fake-key")
        result = client.send([{"role": "user", "content": "hi"}])
        assert result == "Hello from the model."

    @patch("core.chat.groq_client.Groq")
    def test_passes_messages_and_params_through(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.return_value = make_fake_response()
        client = GroqClient(api_key="fake-key")
        messages = [{"role": "user", "content": "test"}]
        client.send(messages, temperature=0.2, max_tokens=500)
        _, kwargs = mock_instance.chat.completions.create.call_args
        assert kwargs["messages"] == messages
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 500

    @patch("core.chat.groq_client.Groq")
    def test_authentication_error_becomes_friendly_chat_error(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.side_effect = AuthenticationError(
            message="bad key", response=MagicMock(), body=None
        )
        client = GroqClient(api_key="fake-key")
        with pytest.raises(ChatError, match="Invalid Groq API key"):
            client.send([{"role": "user", "content": "hi"}])

    @patch("core.chat.groq_client.Groq")
    def test_rate_limit_error_becomes_friendly_chat_error(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.side_effect = RateLimitError(
            message="rate limited", response=MagicMock(), body=None
        )
        client = GroqClient(api_key="fake-key")
        with pytest.raises(ChatError, match="rate limit"):
            client.send([{"role": "user", "content": "hi"}])

    @patch("core.chat.groq_client.Groq")
    def test_timeout_error_becomes_friendly_chat_error(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        client = GroqClient(api_key="fake-key")
        with pytest.raises(ChatError, match="timed out"):
            client.send([{"role": "user", "content": "hi"}])

    @patch("core.chat.groq_client.Groq")
    def test_connection_error_becomes_friendly_chat_error(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
        client = GroqClient(api_key="fake-key")
        with pytest.raises(ChatError, match="Could not connect"):
            client.send([{"role": "user", "content": "hi"}])

    @patch("core.chat.groq_client.Groq")
    def test_unexpected_exception_becomes_generic_chat_error_not_a_crash(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        mock_instance.chat.completions.create.side_effect = ValueError("something weird")
        client = GroqClient(api_key="fake-key")
        with pytest.raises(ChatError, match="unexpected error"):
            client.send([{"role": "user", "content": "hi"}])

    @patch("core.chat.groq_client.Groq")
    def test_empty_choices_raises_chat_error(self, mock_groq_cls):
        mock_instance = mock_groq_cls.return_value
        empty_response = MagicMock()
        empty_response.choices = []
        mock_instance.chat.completions.create.return_value = empty_response
        client = GroqClient(api_key="fake-key")
        with pytest.raises(ChatError, match="empty response"):
            client.send([{"role": "user", "content": "hi"}])
