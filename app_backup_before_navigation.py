import streamlit as st
import time
from pathlib import Path
import base64
import pandas as pd
import plotly.express as px

from src.csv_processor import load_csv, profile_dataframe, process_mapped_data, CSVProcessorError
from src.ml_pipeline import batch_predict_csv, get_dashboard_kpis

from src.auth import require_login
from src.ml_pipeline import predict_single_customer
from src.llm_agent import generate_action_response

def stream_text(text):
    """Takes fully generated text and visually streams it word-by-word."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03) # Speed of the typing effect

def scroll_to_bottom():
    st.html("""
        <script>
            (function() {
                function scrollEl(el, tag) {
                    if (!el) { console.log('[scroll-debug] ' + tag + ': element not found'); return; }
                    console.log('[scroll-debug] ' + tag + ': scrollHeight=' + el.scrollHeight + ' clientHeight=' + el.clientHeight + ' scrollTop(before)=' + el.scrollTop);
                    try {
                        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                        console.log('[scroll-debug] ' + tag + ': scrollTo called, scrollTop(after)=' + el.scrollTop);
                    } catch (e) {
                        console.log('[scroll-debug] ' + tag + ': scrollTo threw, falling back. Error: ' + e);
                        el.scrollTop = el.scrollHeight;
                    }
                }

                function findScrollable() {
                    var container = document.querySelector('.st-key-campaign_chat_box');
                    console.log('[scroll-debug] container found: ' + (container ? 'yes' : 'NO'));
                    if (!container) return null;
                    if (container.scrollHeight > container.clientHeight) {
                        console.log('[scroll-debug] container itself is scrollable');
                        return container;
                    }
                    var descendants = container.querySelectorAll('*');
                    console.log('[scroll-debug] checking ' + descendants.length + ' descendants for overflow');
                    for (var i = 0; i < descendants.length; i++) {
                        if (descendants[i].scrollHeight > descendants[i].clientHeight) {
                            console.log('[scroll-debug] found scrollable descendant: ' + descendants[i].tagName + '.' + descendants[i].className);
                            return descendants[i];
                        }
                    }
                    console.log('[scroll-debug] no scrollable descendant found, using container as fallback');
                    return container;
                }

                console.log('[scroll-debug] scroll_to_bottom() invoked at ' + Date.now());
                requestAnimationFrame(function() {
                    requestAnimationFrame(function() {
                        scrollEl(findScrollable(), 'main-call');
                    });
                });
            })();
        </script>
    """, unsafe_allow_javascript=True)

def preserve_scroll():
    st.html("""
        <script>
            (function() {
                var mainEl = document.querySelector('section[data-testid="stMain"]');
                if (mainEl) {
                    var saved = sessionStorage.getItem('scrollPos');
                    if (saved !== null) { mainEl.scrollTop = parseInt(saved); }
                    mainEl.addEventListener('scroll', function() {
                        sessionStorage.setItem('scrollPos', mainEl.scrollTop);
                    });
                }
            })();
        </script>
    """, unsafe_allow_javascript=True)

def change_app_mode():
    st.session_state.app_mode = st.session_state.nav_selection

preserve_scroll()

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Marketing Intelligence Platform",
    page_icon="🎯",
    layout="wide"
)

require_login()

# Initialize Session State
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

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================
# 2. HEADER & NAVIGATION
# ==========================================
if not st.session_state.full_screen:
    
    # 1. Base64 Encode Logo
    image_path = Path(__file__).parent / "assets" / "logo.jpeg"
    
    try:
        image_data = base64.b64encode(image_path.read_bytes()).decode()
    except FileNotFoundError:
        # Failsafe just in case the image path is slightly off during testing
        image_data = ""

    # 2. Header Columns (Logo | Title | Logout)
    header_col1, header_col2, header_col3, header_col4 = st.columns([0.06, 0.78, 0.10, 0.06])

    with header_col1:
        if image_data:
            st.markdown(
                f"""
                <img src="data:image/jpeg;base64,{image_data}"
                    class="ai-avengers-logo">
                """,
                unsafe_allow_html=True
            )

    with header_col2:
        st.markdown(
            """
            <h1 style="margin-top: 0; padding-top: 0;" class="ai-title">
                MARKETRON
            </h1>
            <h3 style="margin-top: -10px; color: #64748b;" class="ai-subtitle">
                “Segment Smarter. Market Better.”
            </h3>
            """,
            unsafe_allow_html=True
        )
        
    with header_col3:
        
        if st.button("Logout", width='stretch'):
            st.session_state.clear()
            st.rerun()

    with header_col4:
        # Render the Google Profile picture as a perfect circle
        profile_pic_url = st.session_state.get("user_picture", "")
        if profile_pic_url:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end;">
                    <img src="{profile_pic_url}" width="42" height="42" style="border-radius: 50%; border: 2px solid #E2E8F0; object-fit: cover;">
                </div>
                """,
                unsafe_allow_html=True
            )

    # 3. Description & Divider
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
# SIDEBAR NAVIGATION
# ==========================================

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Single Customer Analysis"

with st.sidebar:

    # ==========================================
    # MARKETRON BRAND
    # ==========================================

    st.markdown(
        """
        <div style="
            color: #2563EB;
            font-size: 1.45rem;
            font-weight: 750;
            letter-spacing: 1.5px;
            margin: 4px 0 18px 0;
        ">
            MARKETRON
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # USER / LOGIN CARD
    # ==========================================

    with st.container(border=True):

        st.success("✅ Logged in")

        user_name = getattr(st.user, "name", "User")

        st.markdown(
            f"""
            <div style="
                font-size: 0.95rem;
                color: #334155;
                margin: 4px 0 12px 0;
            ">
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

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # NAVIGATION
    # ==========================================

    nav_options = [
        "Single Customer Analysis",
        "Batch CSV Processor"
    ]

    with st.container(key="sidebar_navigation"):

        st.radio(
            "Navigation",
            options=nav_options,
            key="nav_selection",
            index=nav_options.index(st.session_state.app_mode),
            on_change=change_app_mode,
            label_visibility="collapsed"
        )


# ==========================================
# APP ROUTING ENGINE
# ==========================================

if st.session_state.app_mode == "Single Customer Analysis":
    # ==========================================
    # 3. BLOCK 1: INPUT BANNER
    # ==========================================
    predict_btn = False
    if not st.session_state.full_screen:
        with st.container(key="cohort_banner"):
            banner_left, banner_right = st.columns([1.2, 1], gap="large")

            with banner_left:
                st.markdown("<h3 style='color: #38BDF8; margin-top: -10px;'>Segment Profile Input</h3>", unsafe_allow_html=True)
                st.write("Provide the **average engagement metrics** for this customer segment:")
            
                with st.form("customer_input_form"):
                    top_cat = st.text_input(
                        "Primary Product Category",
                        placeholder="e.g., Wireless Earbuds, Coffee Beans"
                    )
                
                    c1, c2 = st.columns(2)
                    with c1:
                        recency = st.number_input("Average Recency (Days)", min_value=0, max_value=1000, value=25)
                        monetary = st.number_input("Average Total Spend", min_value=1000.0, max_value=100000.0, value=1000.0, step=500.0)
                    with c2:
                        frequency = st.number_input("Average Frequency (Orders)", min_value=1, max_value=200, value=3)
                        # Swapped out Review Score for Price Sensitivity
                        price_sensitivity = st.selectbox(
                            "Price Sensitivity",
                            options=["Full Price Consumers", "Bargain Hunters", "Seasonal Shoppers"]
                        )
                    
                    predict_btn = st.form_submit_button("🔍 Analyze Segment Metrics",key="analyze_btn", width='stretch')

            with banner_right:
                st.markdown("""
                    <div style="padding-top: 20px;">
                        <h1 style="color: #38BDF8; margin-bottom: 15px; font-family: sans-serif; letter-spacing: 3px;">Segment-Level Intelligence</h1>
                        <p style="font-size: 1.05em; line-height: 1.6; color: #CBD5E1; margin-bottom: 20px;">
                            Provide average customer behavior metrics to instantly identify their segment and get an AI-crafted marketing approach built specifically for that group.
                        </p>
                    </div>
                """, unsafe_allow_html=True) 

    # ==========================================
    # 4. BLOCK 2: CENTRILIZED PROCESSING & CHAT 
    # ==========================================

    # 1. Centered Loading Animation 
    if predict_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.status("🧠 Processing Segment Intelligence...", expanded=True) as status:
            st.write("Classifying segment via ML Matrix...")
            time.sleep(0.4) 
            
            try:
                # 1. Removed review_score; passing pure RFM to the ML backend
                pred = predict_single_customer(recency, frequency, monetary)
                
                if "Champions" in pred["segment_name"]: badge_color = "green"
                elif "Risk" in pred["segment_name"]: badge_color = "red"
                elif "Churned" in pred["segment_name"]: badge_color = "gray"
                else: badge_color = "blue"

                st.session_state.current_segment = pred["segment_name"]
                st.session_state.top_category = top_cat if top_cat.strip() != "" else "General Merchandise"
                
                # 2. Saving the new dropdown value to session state for LLM injection
                st.session_state.price_sensitivity = price_sensitivity
                
                st.session_state.segment_desc = pred["description"]
                st.session_state.badge_color = badge_color
                st.session_state.chat_history = [] 
                
                st.write("Crafting your AI-powered campaign approach...")
                initial_prompt = "Provide a brief, high-level overview of exactly how we should market to this specific segment."
                _, st.session_state.chat_history = generate_action_response(
                    st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, st.session_state.price_sensitivity,initial_prompt
                )
                
                # HIDDEN FLAG: Hide the automated initial prompt from the UI
                st.session_state.chat_history[-2]["hidden"] = True
                
                st.session_state.stream_latest = True 
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Analysis Failed", state="error", expanded=False)
                st.error(f"Backend Error: {e}")  

    # 2. Full Width Content Layout
    if st.session_state.current_segment:
        
        st.markdown("<br>", unsafe_allow_html=True)

        # ---> HIDE THESE BLOCKS IN FULL SCREEN <---
        if not st.session_state.full_screen:
            st.markdown(f"""
                <div style="background-color: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;">
                    <span style="color: #065F46; font-size: 0.95em; font-weight: 500;">Segment Match</span><br>
                    <span style="color: #047857; font-size: 1.8em; font-weight: 800; letter-spacing: 0.5px;">{st.session_state.current_segment}</span>
                </div>
            """, unsafe_allow_html=True)
        
            st.markdown(f"""
                <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 14px 20px; margin-bottom: 10px;">
                    <span style="color: #1E40AF; font-weight: 600;">Category:</span>
                    <code>{st.session_state.top_category}</code> | {st.session_state.segment_desc}
                </div>
            """, unsafe_allow_html=True)
        
            st.markdown("---")

        # Wrap buttons in a scoped container to color-code them individually
        with st.container(key="action_container"):
            
            # ⛶ FULL SCREEN NAV BAR
            nav_c1, nav_c2 = st.columns([5, 1])
            with nav_c1:
                if st.session_state.full_screen:
                    if st.button("⬅️ Back to Dashboard",key="back_btn" ,width='content'):
                        st.session_state.full_screen = False
                        st.rerun()
            with nav_c2:
                if not st.session_state.full_screen:
                    if st.button("⛶ Full Screen",key="fullscreen_btn", width='stretch'):
                        st.session_state.full_screen = True
                        st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ACTION BUTTONS (NOW ACTING AS STATE TRIGGERS)
            btn_c1, btn_c2, btn_c3 = st.columns(3)
            
            with btn_c1:
                if st.button("📧 Email Draft", key="email_btn", width='stretch'):
                    st.session_state.pending_prompt = "Write a high-converting marketing email template (with a subject line) for this segment. Give them a compelling reason to buy again today."
                    st.session_state.pending_action = "Drafting Email Campaign..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #991B1B; font-size: 0.85em; font-weight: 500;">Direct-response template for immediate conversions.</span>
                    </div>
                """, unsafe_allow_html=True)
                        
            with btn_c2:
                if st.button("📢 Social Ad Copy", key="ad_btn", width='stretch'):
                    st.session_state.pending_prompt = "Draft short, punchy Facebook/Instagram Ad copy for this segment. Include a Headline, Body Text, and CTA."
                    st.session_state.pending_action = "Drafting Ad Campaign..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #92400E; font-size: 0.85em; font-weight: 500;">Scroll-stopping social media ad creative.</span>
                    </div>
                """, unsafe_allow_html=True)
                        
            with btn_c3:
                if st.button("🎯 Retention Strategy", key="strategy_btn", width='stretch'):
                    st.session_state.pending_prompt = "Provide a detailed, bulleted 3-step retention strategy and follow-up sequence for this segment."
                    st.session_state.pending_action = "Building Strategy..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #1E40AF; font-size: 0.85em; font-weight: 500;">Step-by-step engagement and retention plan.</span>
                    </div>
                """, unsafe_allow_html=True)
        
        # ==========================================
        # 5. INTELLIGENT CHAT ENGINE
        # ==========================================
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chat box always has a fixed scroll height — taller in fullscreen since it's the main content there
        chat_height = 650 if st.session_state.full_screen else 500
        chat_box = st.container(height=chat_height, key="campaign_chat_box")
        
        with chat_box:
            if len(st.session_state.chat_history) == 0:
                st.caption("Click a campaign button above or use the custom prompt menu to start drafting.")
            
            # Render past history
            for msg in st.session_state.chat_history:
                if msg["role"] == "system" or msg.get("hidden", False):
                    continue
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if st.session_state.scroll_pending:
            scroll_to_bottom()
            st.session_state.scroll_pending = False

            # PROMPT INPUT (Pins to the bottom of the screen, outside the container)
        chat_prompt = st.chat_input("Refine this campaign (e.g., 'Make it more urgent', 'Add a 20% discount')...")

        pending_prompt = st.session_state.get("pending_prompt")

        # 4. EXECUTION & IN-LINE LOADING
        if chat_prompt or pending_prompt:
            prompt = chat_prompt if chat_prompt else pending_prompt
            action_text = st.session_state.get("pending_action", "Refining Campaign...")

            # Button prompts stay hidden, typed prompts render in the UI
            is_hidden = True if pending_prompt else False
            st.session_state.chat_history.append({"role": "user", "content": prompt, "hidden": is_hidden})

            # 🛠️ THE FIX: Force the generation UI to render INSIDE the scrollable chat box
            with chat_box:
                
                if not is_hidden:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                scroll_to_bottom()

                # Render the loading animation INSIDE the AI's chat bubble
                with st.chat_message("assistant"):
                    with st.status(action_text, expanded=True) as status:
                        _, st.session_state.chat_history = generate_action_response(
                            st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, 
                            st.session_state.get("price_sensitivity", "None"),
                            prompt
                        )

                        status.update(label="Complete!", state="complete", expanded=False)

                    st.write_stream(stream_text(st.session_state.chat_history[-1]["content"]))

            st.session_state.scroll_pending = True

            # Clean the triggers and finalize the render
            if "pending_prompt" in st.session_state:
                del st.session_state["pending_prompt"]
            if "pending_action" in st.session_state:
                del st.session_state["pending_action"]
            st.rerun()  

elif st.session_state.app_mode == "Batch CSV Processor":    
    st.markdown("## 📂 Batch Customer Segmentation")
    st.write("Upload a CSV file to analyze thousands of customers at once.")

    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=["csv"])

    if uploaded_file:
        # 1. LOAD & PROFILE
        try:
            raw_df = load_csv(uploaded_file)
            profile = profile_dataframe(raw_df)
            csv_headers = profile["column_names"]
            
            st.success(f"✅ File loaded successfully! ({profile['rows']} rows, {profile['columns']} columns)")
            
            # 2. SELECT PROCESSING MODE
            st.markdown("### ⚙️ Data Configuration")
            mapping_mode = st.radio(
                "What kind of data are you uploading?",
                options=["direct_rfm", "raw_transactions"],
                format_func=lambda x: "📈 Pre-Calculated RFM (Recency, Frequency, Monetary)" if x == "direct_rfm" else "🛒 Raw Transaction Logs (Customer ID, Dates, Spend)"
            )
            
            # 3. STRICT MANUAL COLUMN MAPPING
            st.markdown("### 🗺️ Map Your Columns")
            column_map = {}
            
            if mapping_mode == "direct_rfm":
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    column_map["customer_id"] = st.selectbox("Customer ID", options=csv_headers)
                with c2:
                    column_map["recency"] = st.selectbox("Recency Column", options=csv_headers)
                with c3:
                    column_map["frequency"] = st.selectbox("Frequency Column", options=csv_headers)
                with c4:
                    column_map["monetary"] = st.selectbox("Monetary Column", options=csv_headers)
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    column_map["customer_id"] = st.selectbox("Customer ID", options=csv_headers)
                with c2:
                    column_map["order_date"] = st.selectbox("Order Date", options=csv_headers)
                with c3:
                    column_map["order_id"] = st.selectbox("Order ID", options=csv_headers)
                with c4:
                    column_map["spend"] = st.selectbox("Spend/Price", options=csv_headers)
            
            # 4. EXECUTION & ERROR HANDLING
            if st.button("🚀 Process Batch Data", width='stretch'):
                with st.spinner("Classifying segments..."):
                    try:
                        mapped_df = process_mapped_data(raw_df, mapping_mode, column_map)
                        results_df = batch_predict_csv(mapped_df)
                        kpis = get_dashboard_kpis(results_df)
                        
                        st.session_state.batch_results = results_df
                        st.session_state.batch_kpis = kpis
                        st.session_state.batch_processed = True
                        
                        st.rerun() 
                        
                    except CSVProcessorError as e:
                        st.error(f"⚠️ **Data Ingestion Error:** {e}")
                    except ValueError as e:
                        st.error(f"⚠️ **Data Validation Error:** {e}")
                    except Exception as e:
                        st.error(f"⚠️ **An unexpected error occurred:** {e}")
                        
        except CSVProcessorError as e:
            st.error(f"⚠️ **File Load Error:** {e}")

    # ============================================================
    # BATCH DASHBOARD (Premium Visualizations)
    # ============================================================
    if st.session_state.get("batch_processed"):
        st.markdown("---")
        st.markdown("## 📊 Batch Segmentation Results")
        
        results_df = st.session_state.batch_results
        kpis = st.session_state.batch_kpis
        
        # 1. TOP-LEVEL KPI CARDS
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", f"{kpis['total_customers']:,}")
        c2.metric("Average Spend", f"₹{kpis['average_monetary']:,.2f}")
        c3.metric("Total Segments", len(kpis["segment_distribution"]))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. PIE & BAR CHARTS (Row 1)
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Segment Distribution")
            st.write("Percentage of your total customer base.")
            fig_pie = px.pie(results_df, names="Segment", hole=0.4, color="Segment")
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, width='stretch')
            
        with chart_col2:
            st.markdown("#### Customer Count per Segment")
            st.write("Total volume of customers in each cohort.")
            fig_bar_count = px.histogram(results_df, x="Segment", color="Segment")
            fig_bar_count.update_layout(xaxis_title="", yaxis_title="Number of Customers", showlegend=False)
            st.plotly_chart(fig_bar_count, width='stretch')

        # 3. SPEND & 3D SCATTER (Row 2)
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            st.markdown("#### Average Spend per Segment")
            st.write("Monetary value generated by each cohort.")
            avg_spend_df = results_df.groupby("Segment")["Monetary"].mean().reset_index()
            fig_bar_spend = px.bar(avg_spend_df, x="Segment", y="Monetary", color="Segment")
            fig_bar_spend.update_layout(xaxis_title="", yaxis_title="Avg Spend (₹)", showlegend=False)
            st.plotly_chart(fig_bar_spend, width='stretch')
            
        with chart_col4:
            st.markdown("#### 3D Customer Universe")
            st.write("Interactive map of your customer base (Sampled for speed).")
            sample_df = results_df.sample(min(1000, len(results_df)))
            fig_3d = px.scatter_3d(
                sample_df, x="Recency", y="Frequency", z="Monetary", 
                color="Segment", opacity=0.7, size_max=10
            )
            fig_3d.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_3d, width='stretch')

        st.markdown("---")
        
        # 4. FULL INTERACTIVE DATABASE
        st.markdown("### 📋 Customer Database")
        st.write("Your original data, now enhanced with ML segment predictions. You can sort and filter this table directly.")
        st.dataframe(results_df, width='stretch', height=250)
        
        csv_export = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Segmented CSV",
            data=csv_export,
            file_name="ml_segmented_customers.csv",
            mime="text/csv"
        )

        st.markdown("---")
        
        # ============================================================
        # 5. AI CAMPAIGN GENERATOR (MIRRORED ENGINE)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🤖 Generate Targeted Marketing Strategy")
        st.write("Select a customer segment to instantly generate a hyper-personalized marketing approach using AI.")
        
        available_segments = sorted(results_df["Segment"].unique())
        
        gen_col1, gen_col2 = st.columns([2, 1])
        
        with gen_col1:
            target_segment = st.selectbox(
                "Select Target Segment:", 
                options=available_segments,
                label_visibility="collapsed"
            )
            
        with gen_col2:
            generate_btn = st.button("🚀 Generate Approach", key="batch_generate_btn", width='stretch')
            
        # THE FIX: Run the initial loading widget outside the chatbox, exactly like the main page!
        if generate_btn:
            st.session_state.target_generation_segment = target_segment
            st.session_state.batch_chat_history = []
            
            st.markdown("<br>", unsafe_allow_html=True)
            with st.status(f"🧠 Processing Strategy for {target_segment}...", expanded=True) as status:
                st.write("Connecting to AI Agent...")
                time.sleep(0.4)
                st.write("Crafting your AI-powered campaign approach...")
                
                initial_prompt = "Provide a brief, high-level overview of exactly how we should market to this specific segment."
                _, st.session_state.batch_chat_history = generate_action_response(
                    st.session_state.batch_chat_history, 
                    st.session_state.target_generation_segment, 
                    "General Merchandise", 
                    "None",
                    initial_prompt
                )
                
                # HIDDEN FLAG: Hide the automated initial prompt from the UI
                st.session_state.batch_chat_history[-2]["hidden"] = True
                
                status.update(label="Strategy Generated!", state="complete", expanded=False)
                
            st.session_state.trigger_ai_generation = True

        # Render Strategy & Campaign Workspace
        if st.session_state.get("trigger_ai_generation"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background-color: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 10px; padding: 14px 20px; margin-bottom: 15px;">
                    <span style="color: #065F46; font-size: 0.9em; font-weight: 500;">Active Cohort Target</span><br>
                    <span style="color: #047857; font-size: 1.5em; font-weight: 800;">{st.session_state.target_generation_segment}</span>
                </div>
            """, unsafe_allow_html=True)

            # ACTION BUTTONS CONTAINER
            with st.container(key="batch_action_container"):
                btn_c1, btn_c2, btn_c3 = st.columns(3)
                
                with btn_c1:
                    if st.button("📧 Email Draft", key="batch_email_btn", width='stretch'):
                        st.session_state.batch_pending_prompt = "Write a high-converting marketing email template (with a subject line) for this segment. Give them a compelling reason to buy again today."
                        st.session_state.batch_pending_action = "Drafting Email Campaign..."
                        st.rerun()
                    
                    st.markdown("""
                        <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                            <span style="color: #991B1B; font-size: 0.85em; font-weight: 500;">Direct-response template for immediate conversions.</span>
                        </div>
                    """, unsafe_allow_html=True)
                            
                with btn_c2:
                    if st.button("📢 Social Ad Copy", key="batch_ad_btn", width='stretch'):
                        st.session_state.batch_pending_prompt = "Draft short, punchy Facebook/Instagram Ad copy for this segment. Include a Headline, Body Text, and CTA."
                        st.session_state.batch_pending_action = "Drafting Ad Campaign..."
                        st.rerun()
                    
                    st.markdown("""
                        <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                            <span style="color: #92400E; font-size: 0.85em; font-weight: 500;">Scroll-stopping social media ad creative.</span>
                        </div>
                    """, unsafe_allow_html=True)
                            
                with btn_c3:
                    if st.button("🎯 Retention Strategy", key="batch_strategy_btn", width='stretch'):
                        st.session_state.batch_pending_prompt = "Provide a detailed, bulleted 3-step retention strategy and follow-up sequence for this segment."
                        st.session_state.batch_pending_action = "Building Strategy..."
                        st.rerun()
                    
                    st.markdown("""
                        <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                            <span style="color: #1E40AF; font-size: 0.85em; font-weight: 500;">Step-by-step engagement and retention plan.</span>
                        </div>
                    """, unsafe_allow_html=True)

            # CHAT BOX CONTAINER
            st.markdown("<br>", unsafe_allow_html=True)
            batch_chat_box = st.container(height=500, key="batch_campaign_chat_box")
            
            with batch_chat_box:
                if len(st.session_state.batch_chat_history) == 0:
                    st.caption("Click a campaign button above or use the chat prompt to start drafting.")
                
                for msg in st.session_state.batch_chat_history:
                    if msg["role"] == "system" or msg.get("hidden", False):
                        continue
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if st.session_state.batch_scroll_pending:
                scroll_to_bottom()
                st.session_state.batch_scroll_pending = False

            # CHAT INPUT & PROMPT HANDLING
            batch_chat_prompt = st.chat_input("Refine this campaign (e.g., 'Make it more urgent', 'Focus on reactivation')...", key="batch_chat_input")
            batch_pending_prompt = st.session_state.get("batch_pending_prompt")

            if batch_chat_prompt or batch_pending_prompt:
                prompt = batch_chat_prompt if batch_chat_prompt else batch_pending_prompt
                action_text = st.session_state.get("batch_pending_action", "Refining Campaign...")

                is_hidden = True if batch_pending_prompt else False

                with batch_chat_box:
                    if not is_hidden:
                        with st.chat_message("user"):
                            st.markdown(prompt)

                    scroll_to_bottom()

                    with st.chat_message("assistant"):
                        with st.status(action_text, expanded=True) as status:

                            _, st.session_state.batch_chat_history = generate_action_response(
                                st.session_state.batch_chat_history,
                                
                                st.session_state.target_generation_segment,
                                "General Merchandise",
                                "None",
                                prompt
                            )

                            # THE FIX: Retroactively hide the prompt AFTER the LLM runs.
                            # This guarantees the System Prompt is never bypassed!
                            if is_hidden:
                                st.session_state.batch_chat_history[-2]["hidden"] = True

                            status.update(label="Complete!", state="complete", expanded=False)

                        st.write_stream(stream_text(st.session_state.batch_chat_history[-1]["content"]))

                st.session_state.batch_scroll_pending = True

                if "batch_pending_prompt" in st.session_state:
                    del st.session_state["batch_pending_prompt"]
                if "batch_pending_action" in st.session_state:
                    del st.session_state["batch_pending_action"]
                st.rerun()

        st.markdown("---")
        if st.button("🔄 Upload New File", key="batch_upload_new_btn"):
            st.session_state.batch_processed = False
            st.session_state.trigger_ai_generation = False
            st.session_state.batch_chat_history = []
            st.rerun()
