"""Unit tests for core/chat/prompt_builder.py."""
import json

from core.chat.chat_history import ChatHistory
from core.chat.prompt_builder import PromptBuilder
from models.candidate import Candidate
from models.job_description import JobDescription


class TestSystemPrompt:
    def test_mentions_never_hallucinate_rule(self):
        prompt = PromptBuilder().build_system_prompt()
        assert "invent" in prompt.lower() or "hallucinate" in prompt.lower()

    def test_mentions_never_recalculate_scores_rule(self):
        prompt = PromptBuilder().build_system_prompt()
        assert "recalculate" in prompt.lower() or "never" in prompt.lower()


class TestContextBlock:
    def test_context_excludes_raw_text(self):
        candidate = Candidate(full_name="Jane Doe", raw_text="SECRET FULL RESUME TEXT SHOULD NOT LEAK")
        jd = JobDescription(raw_text="jd", cleaned_text="jd")
        block = PromptBuilder().build_context_block([candidate], jd)
        assert "SECRET FULL RESUME TEXT" not in block

    def test_context_includes_candidate_summary_fields(self):
        candidate = Candidate(full_name="Jane Doe", skills=["Python", "AWS"])
        jd = JobDescription(raw_text="jd", cleaned_text="jd", title="Engineer")
        block = PromptBuilder().build_context_block([candidate], jd)
        assert "Jane Doe" in block
        assert "Python" in block
        assert "Engineer" in block

    def test_context_is_valid_json_after_prefix(self):
        candidate = Candidate(full_name="Jane Doe")
        jd = JobDescription(raw_text="jd", cleaned_text="jd")
        block = PromptBuilder().build_context_block([candidate], jd)
        json_part = block.split("CANDIDATE DATA (JSON):\n", 1)[1]
        parsed = json.loads(json_part)  # must not raise
        assert "candidates" in parsed
        assert "job_description" in parsed

    def test_multiple_candidates_all_included(self):
        candidates = [Candidate(full_name=f"Candidate {i}") for i in range(3)]
        jd = JobDescription(raw_text="jd", cleaned_text="jd")
        block = PromptBuilder().build_context_block(candidates, jd)
        for c in candidates:
            assert c.full_name in block


class TestBuildMessages:
    def test_message_order_system_then_history_then_question(self):
        candidate = Candidate(full_name="Jane Doe")
        jd = JobDescription(raw_text="jd", cleaned_text="jd")
        history = ChatHistory()
        history.add_user_message("previous question")
        history.add_assistant_message("previous answer")

        messages = PromptBuilder().build_messages("new question", [candidate], jd, history)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "system"
        assert messages[2] == {"role": "user", "content": "previous question"}
        assert messages[3] == {"role": "assistant", "content": "previous answer"}
        assert messages[-1] == {"role": "user", "content": "new question"}

    def test_empty_history_still_produces_valid_messages(self):
        candidate = Candidate(full_name="Jane Doe")
        jd = JobDescription(raw_text="jd", cleaned_text="jd")
        messages = PromptBuilder().build_messages("question", [candidate], jd, ChatHistory())
        assert messages[-1]["content"] == "question"
        assert len(messages) == 3  # 2 system + 1 user

    def test_context_rebuilt_fresh_not_accumulated_in_history(self):
        """The data context shouldn't pile up in history across turns —
        only the conversational text should."""
        candidate = Candidate(full_name="Jane Doe")
        jd = JobDescription(raw_text="jd", cleaned_text="jd")
        history = ChatHistory()
        history.add_user_message("first question")
        history.add_assistant_message("first answer")
        messages = PromptBuilder().build_messages("second question", [candidate], jd, history)
        # Only 2 system messages total, regardless of how many turns preceded this one
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 2
