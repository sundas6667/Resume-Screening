"""
ui/pages/chatbot.py  🤖 AI HR Assistant
Chat UI wired to core/chat/chatbot.py. Consumes already-ranked candidates —
this page never recomputes a score or re-runs matching itself.
"""
from __future__ import annotations

import streamlit as st

from ui.components import render_empty_state
from ui.session_state import get_or_create_chatbot, has_results


def render() -> None:
    st.title("🤖 AI HR Assistant")

    if not has_results():
        render_empty_state("🤖", "No candidates yet", "Process resumes from the sidebar to chat about them.")
        return

    if not st.session_state.get("groq_api_key"):
        st.warning("⚠️ Enter a Groq API key in the sidebar to use the AI assistant.")
        return

    bot = get_or_create_chatbot()
    if bot is None:
        st.error("Could not initialize the assistant. Please check your API key in the sidebar.")
        return

    jd_title = bot.job_description.title or "this role"
    st.caption(f"Ready to answer questions about {len(bot.candidates)} candidate(s) for **{jd_title}**.")

    if len(bot.history) == 0:
        st.markdown("**💡 Suggested questions**")
        cols = st.columns(2)
        for i, question in enumerate(bot.suggested_questions()):
            with cols[i % 2]:
                if st.button(question, key=f"suggested_{i}", width="stretch"):
                    _ask(bot, question)
                    st.rerun()

    for message in bot.history.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)

    if question := st.chat_input("Ask about the candidates..."):
        _ask(bot, question)
        st.rerun()

    if len(bot.history) > 0:
        if st.button("🗑️ Clear Chat"):
            bot.clear_history()
            st.rerun()

    if bot.last_retrieval:
        with st.expander("🔎 What informed the last answer", expanded=False):
            diag = bot.last_retrieval
            st.caption(f"Based on {diag.retrieved_count} candidate(s): {', '.join(diag.retrieved_candidate_names)}")
            if diag.average_similarity is not None:
                st.caption(f"Average relevance to question: {diag.average_similarity:.1f}%")


def _ask(bot, question: str) -> None:
    with st.spinner("Thinking..."):
        bot.ask(question)
