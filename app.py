"""
app.py
=======
Entry point: page config, custom theme CSS, session state init, the
persistent sidebar, and page navigation. Every page under ui/pages/ reads
from session_state and calls into core/ , this file just wires it together;
it contains no business logic of its own.
"""
import streamlit as st

from config import APP_NAME
from ui.pages import analytics, candidates, chatbot, dashboard, reports, screening, settings
from ui.session_state import init_session_state
from ui.sidebar import render_sidebar
from ui.styles import inject_custom_css

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
inject_custom_css()
render_sidebar()

pg = st.navigation([
    st.Page(dashboard.render, title="Dashboard", icon="🏠", default=True, url_path="dashboard"),
    st.Page(screening.render, title="Resume Screening", icon="📄", url_path="screening"),
    st.Page(candidates.render, title="Candidates", icon="👥", url_path="candidates"),
    st.Page(analytics.render, title="Analytics", icon="📊", url_path="analytics"),
    st.Page(chatbot.render, title="AI HR Assistant", icon="🤖", url_path="assistant"),
    st.Page(reports.render, title="Reports", icon="📑", url_path="reports"),
    st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
])
pg.run()
