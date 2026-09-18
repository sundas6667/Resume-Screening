"""
ui/styles.py
=============
Injects the extra polish Streamlit's native theming can't achieve  cards,
score badges, hover effects  built entirely from config.py's design
tokens (COLOR_TOKENS/THEME/SPACING/RADIUS/SHADOWS), so the CSS and the
Python-side color logic never drift out of sync. The base dark theme
(colors for buttons/inputs/etc.) is set natively via .streamlit/config.toml.
"""
from __future__ import annotations

import streamlit as st

from config import COLOR_TOKENS, RADIUS, SHADOWS, SPACING, THEME


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --bg-deep: {THEME['background']};
            --bg-surface: {THEME['surface']};
            --bg-surface-alt: {THEME['surface_alt']};
            --border: {THEME['border']};
            --primary: {THEME['primary']};
            --primary-soft: {COLOR_TOKENS['primary']['700']};
            --text: {THEME['text_primary']};
            --muted: {THEME['text_muted']};
            --shadow-md: {SHADOWS['md']};
            --shadow-lg: {SHADOWS['lg']};
            --text-color: #0F172A;
            --secondary-text-color: #475569;
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            background: var(--bg-deep);
            color: var(--text);
        }}

        .stApp {{
            background: var(--bg-deep);
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}

        h1, h2, h3, h4, h5, h6 {{
            letter-spacing: -0.03em;
            color: var(--text) !important;
        }}

        p, label, [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{
            color: var(--text) !important;
        }}

        [data-testid="stCaptionContainer"] {{
            color: #475569 !important;
        }}

        div[data-testid="stMetric"] {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: {RADIUS['xl']}px;
            padding: {SPACING['lg']}px {SPACING['xl']}px;
            box-shadow: var(--shadow-lg);
            transition: all 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary);
        }}

        div[data-testid="stMetric"] > label {{
            color: var(--muted) !important;
            font-weight: 600;
        }}

        div[data-testid="stMetric"] > div {{
            color: var(--text) !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }}

        [data-testid="stVerticalBlockBorderWrapper"],
        .stDataFrame, .stTable, .stExpander {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: {RADIUS['lg']}px;
        }}

        .score-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: {RADIUS['full']}px;
            font-weight: 700;
            font-size: 13px;
            letter-spacing: 0.02em;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
        }}
        .score-badge-excellent {{
            background: linear-gradient(135deg, {COLOR_TOKENS['success']['700']}, {COLOR_TOKENS['success']['500']});
            color: {COLOR_TOKENS['success']['50']};
        }}
        .score-badge-good {{
            background: linear-gradient(135deg, {COLOR_TOKENS['warning']['700']}, {COLOR_TOKENS['warning']['500']});
            color: {COLOR_TOKENS['warning']['50']};
        }}
        .score-badge-poor {{
            background: linear-gradient(135deg, {COLOR_TOKENS['danger']['700']}, {COLOR_TOKENS['danger']['500']});
            color: {COLOR_TOKENS['danger']['50']};
        }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: var(--text);
            font-weight: 700;
        }}

        section[data-testid="stSidebar"] .stButton > button {{
            background: var(--primary) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--primary) !important;
            border-radius: {RADIUS['lg']}px;
            font-weight: 800;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-md);
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            transform: translateY(-1px);
            background: #115E59 !important;
            border-color: #115E59 !important;
            box-shadow: var(--shadow-md);
        }}

        .stButton > button {{
            background: var(--primary) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--primary) !important;
            box-shadow: var(--shadow-md);
        }}
        .stButton > button:hover {{
            background: #115E59 !important;
            color: #FFFFFF !important;
            border-color: #115E59 !important;
        }}
        .stButton > button:focus {{
            color: #FFFFFF !important;
            border-color: #115E59 !important;
            box-shadow: 0 0 0 2px #99F6E4 !important;
        }}

        section[data-testid="stSidebar"] button[kind="secondary"] {{
            background: #FFFFFF !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
        }}

        .stFileUploader > div, .stTextInput > div, .stTextArea > div, .stSelectbox > div {{
            border-radius: {RADIUS['lg']}px;
            border: 1px solid var(--border);
            background: #FFFFFF;
        }}

        .stTextInput input, .stTextArea textarea, .stSelectbox input,
        .stFileUploader section, .stFileUploader small {{
            color: #0F172A !important;
        }}
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
            color: #64748B !important;
            opacity: 1;
        }}

        section[data-testid="stSidebarNav"] a {{
            color: #0F172A !important;
        }}
        section[data-testid="stSidebarNav"] a:hover {{
            background: #F0FDFA !important;
            color: #0F766E !important;
        }}
        section[data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: #CCFBF1 !important;
            color: #0F766E !important;
        }}
        section[data-testid="stSidebarNav"] a[aria-current="page"] span,
        section[data-testid="stSidebarNav"] a[aria-current="page"] svg {{
            color: #0F766E !important;
            fill: currentColor;
        }}

        header[data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: #F8FAFC !important;
        }}
        header[data-testid="stHeader"] button,
        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] a,
        [data-testid="stToolbar"] span {{
            color: #0F172A !important;
        }}
        header[data-testid="stHeader"] button:hover,
        [data-testid="stToolbar"] button:hover,
        [data-testid="stToolbar"] a:hover {{
            color: #0F766E !important;
            background: #CCFBF1 !important;
        }}

        .stTabs [role="tablist"] button {{
            border-radius: {RADIUS['md']}px;
            color: var(--muted);
            font-weight: 600;
        }}

        .stTabs [role="tablist"] button[aria-selected="true"] {{
            background: #CCFBF1;
            color: #0F766E;
            border-bottom: 2px solid var(--primary);
        }}

        .app-empty-state {{
            text-align: center;
            padding: {SPACING['6xl']}px {SPACING['xl']}px;
            border: 1px solid var(--border);
            border-radius: {RADIUS['2xl']}px;
            background: var(--bg-surface);
            box-shadow: var(--shadow-md);
            opacity: 0.98;
        }}
        .app-empty-state-icon {{
            font-size: 54px;
            margin-bottom: {SPACING['lg']}px;
        }}
        .app-empty-state-message {{
            color: {THEME['text_muted']};
            max-width: 520px;
            margin: 0 auto;
            line-height: 1.7;
        }}

        .app-footer {{
            color: {THEME['text_muted']};
            font-size: 11px;
            text-align: center;
            margin-top: 1rem;
            letter-spacing: 0.02em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
