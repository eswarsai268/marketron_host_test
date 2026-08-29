import streamlit as st
import time
import plotly.express as px

from src.csv_processor import (
    load_csv,
    profile_dataframe,
    process_mapped_data,
    CSVProcessorError
)

from src.ml_pipeline import (
    batch_predict_csv,
    get_dashboard_kpis
)

from src.llm_agent import generate_action_response

from src.ui_helpers import (
    stream_text,
    scroll_to_bottom
)

st.markdown("## Batch Customer Segmentation")
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
    st.markdown("## Batch Segmentation Results")
    
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
        fig_pie = px.pie(results_df, names="Segment_Name", hole=0.4, color="Segment_Name")
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, width='stretch')
        
    with chart_col2:
        st.markdown("#### Customer Count per Segment")
        st.write("Total volume of customers in each segment.")
        fig_bar_count = px.histogram(results_df, x="Segment_Name", color="Segment_Name")
        fig_bar_count.update_layout(xaxis_title="", yaxis_title="Number of Customers", showlegend=False)
        st.plotly_chart(fig_bar_count, width='stretch')

    # 3. SPEND & 3D SCATTER (Row 2)
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown("#### Average Spend per Segment")
        st.write("Monetary value generated by each segment.")
        avg_spend_df = results_df.groupby("Segment_Name")["Monetary"].mean().reset_index()
        fig_bar_spend = px.bar(avg_spend_df, x="Segment_Name", y="Monetary", color="Segment_Name")
        fig_bar_spend.update_layout(xaxis_title="", yaxis_title="Avg Spend (₹)", showlegend=False)
        st.plotly_chart(fig_bar_spend, width='stretch')
        
    with chart_col4:
        st.markdown("#### 3D Segment Scatter Plot")
        st.write("Interactive map of your customer base (Sampled for speed).")
        sample_df = results_df.sample(min(1000, len(results_df)))
        fig_3d = px.scatter_3d(
            sample_df, x="Recency", y="Frequency", z="Monetary", 
            color="Segment_Name", opacity=0.7, size_max=10
        )
        fig_3d.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_3d, width='stretch')

    st.markdown("---")
    
    # 4. FULL INTERACTIVE DATABASE
    st.markdown("### Customer Database")
    st.write("Your original data, now enhanced with ML segment predictions. You can sort and filter this table directly.")
    st.dataframe(results_df, width='stretch', height=250)
    
    csv_export = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Segmented CSV",
        data=csv_export,
        file_name="ml_segmented_customers.csv",
        mime="text/csv"
    )

    st.markdown("---")
    
    # ============================================================
    # 5. AI CAMPAIGN GENERATOR (MIRRORED ENGINE)
    # ============================================================
    st.markdown("---")
    st.markdown("### Generate Targeted Marketing Strategy")
    st.write("Select a customer segment to instantly generate a hyper-personalized marketing approach using AI.")
    
    available_segments = sorted(results_df["Segment_Name"].unique())
    
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

        selected_segment_mask = (
            results_df["Segment_Name"] == target_segment
        )

        segment_customer_count = int(
            selected_segment_mask.sum()
        )

        segment_avg_spend = float(
            results_df.loc[
                selected_segment_mask,
                "Monetary"
            ].mean()
        )

        segment_avg_recency = float(
            results_df.loc[
                selected_segment_mask,
                "Recency"
            ].mean()
        )

        segment_avg_frequency = float(
            results_df.loc[
                selected_segment_mask,
                "Frequency"
            ].mean()
        )

        st.session_state.batch_segment_stats = {
            "customer_count": segment_customer_count,
            "avg_spend": segment_avg_spend,
            "avg_recency": segment_avg_recency,
            "avg_frequency": segment_avg_frequency
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.status(f"Processing Strategy for {target_segment}...", expanded=True) as status:
            st.write("Connecting to Marketron AI...")
            time.sleep(0.4)
            st.write("Crafting your AI-powered campaign approach...")
            
            initial_prompt = "Provide a brief, high-level overview of exactly how we should market to this specific segment."
            _, st.session_state.batch_chat_history = generate_action_response(
                st.session_state.batch_chat_history, 
                st.session_state.target_generation_segment, 
                "General Merchandise", 
                "None",
                initial_prompt,
                segment_customer_count=segment_customer_count,
                segment_avg_spend=segment_avg_spend,
                segment_avg_recency=segment_avg_recency,
                segment_avg_frequency=segment_avg_frequency
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
                <span style="color: #065F46; font-size: 0.9em; font-weight: 500;">Active Segment Target</span><br>
                <span style="color: #047857; font-size: 1.5em; font-weight: 800;">{st.session_state.target_generation_segment}</span>
            </div>
        """, unsafe_allow_html=True)

        # ACTION BUTTONS CONTAINER
        with st.container(key="batch_action_container"):
            btn_c1, btn_c2, btn_c3 = st.columns(3)
            
            with btn_c1:
                if st.button("Email Draft", key="batch_email_btn", width='stretch'):
                    st.session_state.batch_pending_prompt = "Write a high-converting marketing email template (with a subject line) for this segment. Give them a compelling reason to buy again today."
                    st.session_state.batch_pending_action = "Drafting Email Campaign..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #991B1B; font-size: 0.85em; font-weight: 500;">Direct-response template for immediate conversions.</span>
                    </div>
                """, unsafe_allow_html=True)
                        
            with btn_c2:
                if st.button("Social Ad Copy", key="batch_ad_btn", width='stretch'):
                    st.session_state.batch_pending_prompt = "Draft short, punchy Facebook/Instagram Ad copy for this segment. Include a Headline, Body Text, and CTA."
                    st.session_state.batch_pending_action = "Drafting Ad Campaign..."
                    st.rerun()
                
                st.markdown("""
                    <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px; text-align: center; margin-top: 5px;">
                        <span style="color: #92400E; font-size: 0.85em; font-weight: 500;">Scroll-stopping social media ad creative.</span>
                    </div>
                """, unsafe_allow_html=True)
                        
            with btn_c3:
                if st.button("Retention Strategy", key="batch_strategy_btn", width='stretch'):
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
                        segment_stats = st.session_state.get(
                            "batch_segment_stats",
                            {}
                        )

                        _, st.session_state.batch_chat_history = generate_action_response(
                            st.session_state.batch_chat_history,
                            
                            st.session_state.target_generation_segment,
                            "General Merchandise",
                            "None",
                            prompt,
                            segment_customer_count=segment_stats.get(
                                "customer_count"
                            ),
                            segment_avg_spend=segment_stats.get(
                                "avg_spend"
                            ),
                            segment_avg_recency=segment_stats.get(
                                "avg_recency"
                            ),
                            segment_avg_frequency=segment_stats.get(
                                "avg_frequency"
                            )
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