"""Unit tests for core/chat/chat_history.py."""
from core.chat.chat_history import ChatHistory, ChatMessage


class TestChatHistory:
    def test_starts_empty(self):
        assert len(ChatHistory()) == 0

    def test_add_user_message(self):
        history = ChatHistory()
        history.add_user_message("Who is the best candidate?")
        assert len(history) == 1
        assert history.messages[0].role == "user"
        assert history.messages[0].content == "Who is the best candidate?"

    def test_add_assistant_message(self):
        history = ChatHistory()
        history.add_assistant_message("John Anderson is the strongest match.")
        assert history.messages[0].role == "assistant"

    def test_messages_returns_copy_not_reference(self):
        history = ChatHistory()
        history.add_user_message("test")
        messages = history.messages
        messages.append(ChatMessage(role="user", content="injected"))
        assert len(history) == 1  # original unaffected

    def test_to_groq_format_shape(self):
        history = ChatHistory()
        history.add_user_message("question")
        history.add_assistant_message("answer")
        formatted = history.to_groq_format()
        assert formatted == [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

    def test_recent_respects_max_messages(self):
        history = ChatHistory(max_messages=2)
        for i in range(5):
            history.add_user_message(f"msg{i}")
        assert len(history.to_groq_format()) == 2
        assert history.to_groq_format()[0]["content"] == "msg3"

    def test_full_history_retained_beyond_max_messages(self):
        history = ChatHistory(max_messages=2)
        for i in range(5):
            history.add_user_message(f"msg{i}")
        assert len(history.messages) == 5  # nothing discarded, just bounded for the API call

    def test_clear_empties_history(self):
        history = ChatHistory()
        history.add_user_message("test")
        history.clear()
        assert len(history) == 0

    def test_recent_with_explicit_n(self):
        history = ChatHistory(max_messages=20)
        for i in range(5):
            history.add_user_message(f"msg{i}")
        assert len(history.recent(n=3)) == 3
