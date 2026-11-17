"""
Retrieval-Augmented Academic Regulations Assistant — Streamlit app.
"""

import json
import sys
from html import escape
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_chain import AcademicRegulationsRAGChain

st.set_page_config(
    page_title="Academic Regulations Assistant",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- theme state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


def on_dark_mode_toggle() -> None:
    st.session_state.dark_mode = st.session_state.dark_mode_switch


# ---------------------------------------------------------------- styles & fonts
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cossette+Titre:wght@400;700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

css_path = PROJECT_ROOT / "static" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Theme overrides (Seamless Light / Dark Mode)
if not st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"] {
            --bg: #FFFFFF !important;
            --alt: #F8F8F8 !important;
            --hover: #EFEFEF !important;
            --line: #E5E5E5 !important;
            --line-subtle: #EEEEEE !important;
            --ink: #0A0A0A !important;
            --ink-2: #525252 !important;
            --ink-3: #8C8C8C !important;
            background-color: var(--bg) !important;
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] {
            background-color: var(--alt) !important;
            border-right-color: var(--line) !important;
        }
        /* Light Mode Toggle Switch */
        [data-testid="stCheckbox"] div[class*="e1hn2pbo7"],
        [data-testid="stCheckbox"] label > div > div:first-child,
        .st-key-dark_mode_switch div[class*="e1hn2pbo7"],
        .st-key-dark_mode_switch label > div > div:first-child {
            background-color: #E2E2E2 !important;
            background: #E2E2E2 !important;
            border: 1px solid #D0D0D0 !important;
        }
        [data-testid="stCheckbox"] div[class*="e1hn2pbo8"],
        [data-testid="stCheckbox"] label > div > div:first-child > div,
        .st-key-dark_mode_switch div[class*="e1hn2pbo8"],
        .st-key-dark_mode_switch label > div > div:first-child > div {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        }
        [data-testid="stChatInput"] {
            background-color: var(--bg) !important;
            border-color: var(--line) !important;
        }
        [data-testid="stChatInputSubmitButton"] {
            background-color: #0A0A0A !important;
        }
        [data-testid="stChatInputSubmitButton"] svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            background: var(--alt) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [data-testid="stExpander"],
        [data-testid="stExpander"] details summary,
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background: var(--alt) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] *,
        [data-baseweb="select"] div,
        [data-baseweb="select"] *,
        [data-baseweb="popover"],
        [data-baseweb="popover"] div,
        [data-baseweb="menu"],
        [data-baseweb="menu"] div,
        ul[role="listbox"],
        li[role="option"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #0A0A0A !important;
            border-color: #E5E5E5 !important;
        }
        .citation-badge {
            background: var(--alt) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        .refusal-alert-card {
            background: var(--alt) !important;
            border-left-color: var(--ink) !important;
            color: var(--ink) !important;
        }
        .query-plan-card {
            border-left-color: var(--line) !important;
            color: var(--ink-2) !important;
        }
        table, thead th, tbody td {
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        thead th {
            background: var(--alt) !important;
        }
        tbody tr:hover {
            background: var(--hover) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"] {
            --bg: #111111 !important;
            --alt: #191919 !important;
            --hover: #242424 !important;
            --line: #2C2C2C !important;
            --line-subtle: #222222 !important;
            --ink: #EDEDED !important;
            --ink-2: #A3A3A3 !important;
            --ink-3: #737373 !important;
            background-color: var(--bg) !important;
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] {
            background-color: var(--alt) !important;
            border-right-color: var(--line) !important;
        }
        [data-testid="stSidebar"] * {
            color: var(--ink);
        }
        /* Dark Mode Toggle Switch (Sleek charcoal track, bright clean thumb) */
        [data-testid="stCheckbox"] div[class*="e1hn2pbo7"],
        [data-testid="stCheckbox"] label > div > div:first-child,
        .st-key-dark_mode_switch div[class*="e1hn2pbo7"],
        .st-key-dark_mode_switch label > div > div:first-child {
            background-color: #2E2E2E !important;
            background: #2E2E2E !important;
            border: 1px solid #3E3E3E !important;
        }
        [data-testid="stCheckbox"] div[class*="e1hn2pbo8"],
        [data-testid="stCheckbox"] label > div > div:first-child > div,
        .st-key-dark_mode_switch div[class*="e1hn2pbo8"],
        .st-key-dark_mode_switch label > div > div:first-child > div {
            background-color: #EDEDED !important;
            background: #EDEDED !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
        }
        [data-testid="stChatInput"] {
            background-color: var(--bg) !important;
            border-color: var(--line) !important;
        }
        [data-testid="stChatInputSubmitButton"] {
            background-color: #EDEDED !important;
        }
        [data-testid="stChatInputSubmitButton"] svg {
            color: #111111 !important;
            fill: #111111 !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            background: var(--alt) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        /* Settings Expander & Selectbox in Dark Mode: ZERO White Boxes */
        [data-testid="stExpander"],
        [data-testid="stExpander"] details summary,
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background: var(--bg) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [data-testid="stSelectbox"],
        [data-testid="stSelectbox"] *,
        [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] > div,
        [data-baseweb="select"],
        [data-baseweb="select"] *,
        [data-baseweb="select"] div,
        [data-baseweb="popover"],
        [data-baseweb="popover"] div,
        [data-baseweb="menu"],
        [data-baseweb="menu"] div,
        ul[role="listbox"],
        li[role="option"] {
            background-color: #191919 !important;
            background: #191919 !important;
            border-color: #2C2C2C !important;
            color: #EDEDED !important;
        }
        [data-baseweb="select"] span,
        [data-baseweb="select"] svg {
            color: #EDEDED !important;
            fill: #EDEDED !important;
        }
        li[role="option"] {
            background-color: transparent !important;
            color: #EDEDED !important;
        }
        li[role="option"]:hover {
            background-color: #242424 !important;
        }
        .citation-badge {
            background: var(--alt) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        .refusal-alert-card {
            background: var(--alt) !important;
            border-left-color: var(--ink) !important;
            color: var(--ink) !important;
        }
        .query-plan-card {
            border-left-color: var(--line) !important;
            color: var(--ink-2) !important;
        }
        table, thead th, tbody td {
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        thead th {
            background: var(--alt) !important;
        }
        tbody tr:hover {
            background: var(--hover) !important;
        }
        pre {
            background: var(--bg) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- state
VIEWS = ["Chat", "Text splitting", "Evaluation"]
QUICK_PROMPTS = [
    "What is the minimum attendance required for ESA, and what happens if I am detained under NSAR?",
    "What are the passing marks in ISA and ESA?",
    "How does the grading work?",
    "What are the requirements for B.Tech with Honours?",
    "What is the penalty for possessing unauthorized notes in an exam?",
    "What are the lunch hours and menu at the university hostel mess?",
]

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = AcademicRegulationsRAGChain(backend="faiss")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "view" not in st.session_state:
    st.session_state.view = "Chat"


def ask(q: str) -> None:
    st.session_state.user_prompt = q
    st.session_state.view = "Chat"


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.rag_chain.memory.clear()
    st.session_state.view = "Chat"


def render_assistant(msg: dict) -> None:
    plan = msg.get("plan") or {}
    if plan.get("is_multipart"):
        subs = plan["sub_queries"]
        items = "".join(
            f'<div class="subquery-item"><span class="subquery-number">{i}</span>'
            f"<span>{escape(sq)}</span></div>"
            for i, sq in enumerate(subs, 1)
        )
        st.markdown(
            f'<div class="query-plan-card"><div class="query-plan-title">'
            f"Query Decomposition ({len(subs)} Sub-questions)</div>{items}</div>",
            unsafe_allow_html=True,
        )
    if msg.get("refused"):
        st.markdown(
            '<div class="refusal-alert-card">'
            '<strong>🛡️ Out of Scope</strong><br>'
            "I decline to answer: The provided university academic regulations do not contain information to support this question."
            '</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(msg["content"])
    pages = msg.get("cited_pages") or []
    if pages:
        badges = "".join(f'<span class="citation-badge">Page {p}</span>' for p in sorted(pages))
        st.markdown(f'<div class="citation-container"><span style="margin-right:2px;">Sources</span>{badges}</div>', unsafe_allow_html=True)


def handle_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the regulations…"):
            res = st.session_state.rag_chain.answer_query(prompt)
        msg = {
            "role": "assistant",
            "content": res["answer"],
            "plan": res["plan"],
            "cited_pages": res["cited_pages"],
            "refused": res["refused"],
        }
        render_assistant(msg)
    st.session_state.messages.append(msg)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    # 1. Clean, Standout Brand Logo (No dots, no badges)
    st.markdown('<div class="brand">ara</div>', unsafe_allow_html=True)

    # 2. Sleek Dark mode toggle
    st.toggle(
        "Dark mode",
        value=st.session_state.dark_mode,
        key="dark_mode_switch",
        on_change=on_dark_mode_toggle,
    )

    # 3. New Chat Button
    st.button("＋  New chat", key="new_chat", on_click=new_chat, use_container_width=True)

    # 4. Suggested Questions with subtle borders (Zero extra gap)
    st.markdown('<div class="side-label">Suggested questions</div>', unsafe_allow_html=True)
    for i, q in enumerate(QUICK_PROMPTS):
        st.button(q, key=f"quick_{i}", on_click=ask, args=(q,), use_container_width=True)

    # 5. Push Settings Below the Fold (Hidden during demo, visible only on deep scroll)
    st.markdown('<div style="height: 48vh;"></div>', unsafe_allow_html=True)

    with st.expander("⚙️ Settings"):
        # View switcher hidden in Settings dropdown alongside vector store
        current_view_idx = VIEWS.index(st.session_state.view) if st.session_state.view in VIEWS else 0
        view_choice = st.selectbox(
            "View",
            VIEWS,
            index=current_view_idx,
            key="view_selector",
        )
        if view_choice != st.session_state.view:
            st.session_state.view = view_choice
            st.rerun()

        current_backend_idx = 0 if st.session_state.rag_chain.backend == "faiss" else 1
        backend_choice = st.selectbox(
            "Vector store",
            ["FAISS", "ChromaDB"],
            index=current_backend_idx,
            key="backend_selector",
        )
        if backend_choice.lower() != st.session_state.rag_chain.backend:
            st.session_state.rag_chain.set_backend(backend_choice.lower())
            st.toast(f"Switched backend to {backend_choice}")

        st.caption("Conversation memory")
        st.text(st.session_state.rag_chain.memory.get_memory_context())


# ---------------------------------------------------------------- main view (NO top tabs)
view = st.session_state.view

if view == "Chat":
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="hero-title">Academic Regulations Assistant</div>
                <div class="hero-subtitle">Ask about credits, attendance, examinations, grading, and academic policies.
                Answers are grounded in the official handbook and cite the page.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Clean message rendering (No DeltaGenerator magic dumps)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant(msg)
            else:
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask about attendance, credits, exams, ISA or grading")
    if "user_prompt" in st.session_state:
        prompt = st.session_state.pop("user_prompt")
    if prompt:
        handle_prompt(prompt)

elif view == "Text splitting":
    st.subheader("Deliverable 1: Text-Splitting Strategy Comparison")
    st.write("Fixed-size, recursive, and semantic chunking evaluated across the official regulations handbook.")
    chart = Path("results/chunking_comparison.png")
    if chart.exists():
        st.image(str(chart), caption="Chunk length distribution by strategy", use_container_width=True)
    stats_file = Path("results/chunking_comparison.json")
    if stats_file.exists():
        with st.expander("Raw Splitting Metrics"):
            st.json(json.loads(stats_file.read_text(encoding="utf-8")))

else:
    st.subheader("Deliverable 3: 50-Question Benchmark Evaluation")
    chart = Path("results/evaluation_charts.png")
    if chart.exists():
        st.image(str(chart), caption="Recall@k, answer quality, and refusal confusion matrix", use_container_width=True)
    report = Path("results/evaluation_report.md")
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
