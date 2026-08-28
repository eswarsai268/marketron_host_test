import streamlit as st
import base64
from pathlib import Path

from src.auth import require_login
from src.ui_helpers import preserve_scroll
from src.rag_engine import start_rag_warmup

st.set_page_config(
    page_title="Marketron",
    page_icon="🎯",
    layout="wide"
)

# ==========================================
# PAGE DEFINITIONS
# ==========================================

segment_page = st.Page(
    "pages/segment_analyzer.py",
    title="Segment Analyzer",
    default=True
)

batch_page = st.Page(
    "pages/batch_segmentation.py",
    title="Batch Segmentation"
)

host_page = st.Page(
    "pages/host.py",
    title="Host"
)

collaborative_chat_page = st.Page(
    "pages/collaborative_chat.py",
    title="Collaborative Chat"
)

# NAVIGATION ROUTER
pg = st.navigation(
    [segment_page, batch_page, host_page,collaborative_chat_page],
    position="hidden"
)



# AUTHENTICATION
require_login()

# ==========================================
# SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_segment" not in st.session_state:
    st.session_state.current_segment = None

if "top_category" not in st.session_state:
    st.session_state.top_category = None

if "stream_latest" not in st.session_state:
    st.session_state.stream_latest = False

if "full_screen" not in st.session_state:
    st.session_state.full_screen = False

if "scroll_pending" not in st.session_state:
    st.session_state.scroll_pending = False

if "batch_chat_history" not in st.session_state:
    st.session_state.batch_chat_history = []

if "batch_scroll_pending" not in st.session_state:
    st.session_state.batch_scroll_pending = False

# GLOBAL CSS
with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# SHARED SCROLL BEHAVIOR
preserve_scroll()

# ==========================================
# SHARED HEADER
# ==========================================
if not st.session_state.full_screen:

    # Logo
    image_path = (
        Path(__file__).parent
        / "assets"
        / "logo.jpeg"
    )

    try:
        image_data = (
            base64
            .b64encode(image_path.read_bytes())
            .decode()
        )
    except FileNotFoundError:
        image_data = ""

    # Header layout
    header_col1, header_col2, header_col3, header_col4 = st.columns(
        [0.06, 0.78, 0.10, 0.06]
    )

    # Logo
    with header_col1:
            if image_data:
                st.markdown(
                    f"""
                    <img src="data:image/jpeg;base64,{image_data}"
                        class="ai-avengers-logo">
                    """,
                    unsafe_allow_html=True
                )

    # Marketron title
    with header_col2:

        st.markdown(
            """
            <h1
                style="
                    margin-top: 0;
                    padding-top: 0;
                "
                class="ai-title"
            >
                MARKETRON
            </h1>
            <h3
                style="
                    margin-top: -10px;
                    color: #64748b;
                "
                class="ai-subtitle"
            >
                “Segment Smarter. Market Better.”
            </h3>
            """,
            unsafe_allow_html=True
        )

    # Header logout
    with header_col3:

        if st.button(
            "Logout",
            key="header_logout",
            width="stretch"
        ):
            st.session_state.clear()
            st.logout()

    # Google profile picture
    with header_col4:

        profile_pic_url = st.session_state.get(
            "user_picture",
            ""
        )

        if profile_pic_url:

            st.markdown(
                f"""
                <div
                    style="
                        display: flex;
                        justify-content: flex-end;
                    "
                >
                    <img
                        src="{profile_pic_url}"
                        width="42"
                        height="42"
                        style="
                            border-radius: 50%;
                            border: 2px solid #E2E8F0;
                            object-fit: cover;
                        "
                    >
                </div>
                """,
                unsafe_allow_html=True
            )

    # Project introduction
    st.markdown(
        """
        <div class="project-intro">
            <h2 class="project-title">
                Customer Segmentation & Personalized Marketing Intelligence
            </h2>
            <p class="project-description">
                Predict customer segments based on behavioral data to unlock
                personalized insights and determine the exact marketing approach
                for each specific group.
            </p>
        </div>
        <hr style="border: 0; border-top: 1px solid #E2E8F0; margin-bottom: 30px;">
        """,
        unsafe_allow_html=True
    ) 

# ==========================================
# CUSTOM SIDEBAR
# ==========================================

with st.sidebar:
    # Marketron branding
    st.markdown(
        """
        <div
            style="
                color: #2563EB;
                font-size: 1.45rem;
                font-weight: 750;
                letter-spacing: 1.5px;
                margin: 4px 0 18px 0;
            "
        >
            MARKETRON
        </div>
        """,
        unsafe_allow_html=True
    )

    # User account card
    with st.container(border=True):
        st.success("Logged in")

        user_name = getattr(
            st.user,
            "name",
            "User"
        )
        st.markdown(
            f"""
            <div
                style="
                    font-size: 0.95rem;
                    color: #334155;
                    margin: 4px 0 12px 0;
                "
            >
                Welcome, <strong>{user_name}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Logout",
            key="sidebar_logout",
            width="stretch"
        ):
            st.session_state.clear()
            st.logout()

    # Navigation links
    st.markdown(
        "<div style='height: 14px;'></div>",
        unsafe_allow_html=True
    )

    st.page_link(
        segment_page,
        label="Segment Analyzer",
        width="stretch"
    )

    st.page_link(
        batch_page,
        label="Batch Segmentation",
        width="stretch"
    )

    st.page_link(
        host_page,
        label="Host",
        width="stretch"
    )

    if st.session_state.get("collaboration_available", False):
        st.page_link(
            collaborative_chat_page,
            label="Collaborative Chat",
            width="stretch"
        )

pg.run()

start_rag_warmup()