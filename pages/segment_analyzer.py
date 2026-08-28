import streamlit as st
import time
if "full_screen" not in st.session_state:
    st.session_state.full_screen = False

if "scroll_pending" not in st.session_state:
    st.session_state.scroll_pending = False

from src.ml_pipeline import predict_single_customer
from src.llm_agent import generate_action_response
from src.ui_helpers import stream_text, scroll_to_bottom

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
                    monetary = st.number_input("Average Total Spend", min_value=1000.0, max_value=1000000.0, value=1000.0, step=1000.0)
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
                st.session_state.chat_history, st.session_state.current_segment, st.session_state.top_category, st.session_state.price_sensitivity,initial_prompt,
                segment_avg_recency=recency,
                segment_avg_frequency=frequency,
                segment_avg_spend=monetary
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
                        prompt,
                        segment_avg_recency=recency,
                        segment_avg_frequency=frequency,
                        segment_avg_spend=monetary
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