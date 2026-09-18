"""Unit tests for core/chat/chatbot.py  injects a fake GroqClient so no
real network call is ever made.
"""
from core.chat.chat_history import ChatHistory
from core.chat.chatbot import HRChatbot
from core.chat.groq_client import ChatError
from models.candidate import Candidate
from models.job_description import JobDescription


class FakeGroqClient:
    """Stand-in for GroqClient  records calls, returns a canned or
    configurable response instead of hitting the network.
    """
    def __init__(self, response: str = "This is a test answer.", raise_error: Exception = None):
        self.response = response
        self.raise_error = raise_error
        self.last_messages = None
        self.call_count = 0

    def send(self, messages, temperature=None, max_tokens=None):
        self.call_count += 1
        self.last_messages = messages
        if self.raise_error:
            raise self.raise_error
        return self.response


def make_chatbot(candidates=None, response="Test answer.", raise_error=None) -> HRChatbot:
    candidates = candidates if candidates is not None else [Candidate(full_name="Jane Doe", skills=["Python"])]
    jd = JobDescription(raw_text="jd", cleaned_text="jd", title="Engineer", required_skills=["Python"])
    fake_client = FakeGroqClient(response=response, raise_error=raise_error)
    bot = HRChatbot(api_key="unused", candidates=candidates, job_description=jd, client=fake_client)
    return bot


class TestAsk:
    def test_returns_answer_text(self):
        bot = make_chatbot(response="John is the best fit.")
        answer = bot.ask("Who is the best candidate?")
        assert answer == "John is the best fit."

    def test_empty_question_returns_prompt_without_calling_api(self):
        bot = make_chatbot()
        answer = bot.ask("")
        assert "ask a question" in answer.lower()
        assert bot.client.call_count == 0

    def test_whitespace_only_question_does_not_call_api(self):
        bot = make_chatbot()
        bot.ask("   ")
        assert bot.client.call_count == 0

    def test_records_question_and_answer_in_history(self):
        bot = make_chatbot(response="The answer.")
        bot.ask("What is the question?")
        assert len(bot.history) == 2
        assert bot.history.messages[0].content == "What is the question?"
        assert bot.history.messages[1].content == "The answer."

    def test_chat_error_returns_friendly_message_not_raised(self):
        bot = make_chatbot(raise_error=ChatError("Invalid Groq API key."))
        answer = bot.ask("Who is best?")
        assert "Invalid Groq API key" in answer
        # A failed call shouldn't pollute history with a broken exchange
        assert len(bot.history) == 0

    def test_does_not_recompute_candidate_scores(self):
        """The chatbot must consume existing RankingResult, never recompute it."""
        candidate = Candidate(full_name="Jane Doe", skills=["Python"])
        candidate.ranking.overall_score = 88.0
        bot = make_chatbot(candidates=[candidate])
        bot.ask("What's Jane's score?")
        assert candidate.ranking.overall_score == 88.0  # untouched


class TestPromptConsistency:
    def test_context_sent_to_groq_reflects_candidate_data(self):
        candidate = Candidate(full_name="Jane Doe", skills=["Python", "AWS"])
        bot = make_chatbot(candidates=[candidate])
        bot.ask("Tell me about Jane")
        sent_content = " ".join(m["content"] for m in bot.client.last_messages)
        assert "Jane Doe" in sent_content
        assert "Python" in sent_content

    def test_raw_text_never_sent_to_groq(self):
        candidate = Candidate(full_name="Jane Doe", raw_text="CONFIDENTIAL FULL RESUME BODY")
        bot = make_chatbot(candidates=[candidate])
        bot.ask("Tell me about Jane")
        sent_content = " ".join(m["content"] for m in bot.client.last_messages)
        assert "CONFIDENTIAL FULL RESUME BODY" not in sent_content


class TestHistoryManagement:
    def test_clear_history_empties_conversation(self):
        bot = make_chatbot()
        bot.ask("question one")
        bot.clear_history()
        assert len(bot.history) == 0

    def test_second_question_includes_first_turn_in_context(self):
        bot = make_chatbot(response="second answer")
        bot.ask("first question")
        bot.ask("second question")
        roles_and_content = [(m["role"], m["content"]) for m in bot.client.last_messages]
        assert ("user", "first question") in roles_and_content


class TestSuggestedQuestions:
    def test_returns_nonempty_list(self):
        assert len(HRChatbot.suggested_questions()) > 0

    def test_returns_a_copy_not_the_internal_list(self):
        questions = HRChatbot.suggested_questions()
        questions.append("injected")
        assert "injected" not in HRChatbot.suggested_questions()


class TestRetrievalDiagnostics:
    def test_populated_after_successful_ask(self):
        bot = make_chatbot()
        assert bot.last_retrieval is None
        bot.ask("Who is best?")
        assert bot.last_retrieval is not None
        assert bot.last_retrieval.retrieved_count >= 1

    def test_not_populated_on_failed_call(self):
        from core.chat.groq_client import ChatError
        bot = make_chatbot(raise_error=ChatError("boom"))
        bot.ask("Who is best?")
        assert bot.last_retrieval is None

    def test_question_number_increments(self):
        bot = make_chatbot()
        bot.ask("first")
        first_number = bot.last_retrieval.question_number
        bot.ask("second")
        assert bot.last_retrieval.question_number == first_number + 1

    def test_session_id_is_stable_across_questions(self):
        bot = make_chatbot()
        bot.ask("first")
        session_id = bot.session_id
        bot.ask("second")
        assert bot.session_id == session_id

    def test_includes_prompt_and_vector_store_versions(self):
        bot = make_chatbot()
        bot.ask("question")
        assert bot.last_retrieval.prompt_version
        assert bot.last_retrieval.vector_store_version


class TestPromptVersioning:
    def test_prompt_builder_exposes_version(self):
        from core.chat.prompt_builder import PromptBuilder
        assert PromptBuilder.version
