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
st.markdown("## 📂 Batch Customer Segmentation")
st.write("Upload a CSV file to analyze thousands of customers at once.")

    # SMALL UPLOAD BUTTON LEFT SIDE
uploaded_file = st.file_uploader(
        "",
        type=["csv"],
        label_visibility="collapsed"
    )
if uploaded_file :

        try:

            raw_df = load_csv(uploaded_file)
            profile = profile_dataframe(raw_df)

            csv_headers = profile["column_names"]

            st.markdown(f"""
            <div style="
                background:#ecfdf5;
                border:1px solid #6ee7b7;
                padding:16px;
                border-radius:12px;
                margin-top:10px;
                margin-bottom:20px;
            ">
                <h4 style="color:#047857;margin:0;background-color:black!important">
                    ✅ File loaded successfully!
                </h4>
                <p style="margin:0;color:#065f46;">
                    {profile['rows']:,} rows •
                    {profile['columns']} columns
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
<style>
.data-config-title{font-size:38px;font-weight:800;color:#67e8f9;margin-bottom:15px;}
</style>
""", unsafe_allow_html=True)


            st.markdown('<div class="data-config-title">## ⚙️ Data Configuration</div>',unsafe_allow_html=True)

            mapping_mode = st.radio(
                "What kind of data uploading?",
                
                ["direct_rfm", "raw_transactions"],
                horizontal=True,
                format_func=lambda x:
                    "Pre-Calculated RFM(Recency,Frequency,Monetry)"
                    if x == "direct_rfm"
                    else " Raw Transaction Logs(Customer ID,Dates,Spend)"
            )
            

            st.markdown("---")

            st.markdown("##  Map Your Columns")

            column_map = {}

            if mapping_mode == "direct_rfm":

                c1, c2, c3 = st.columns(3)

                with c1:
                    column_map["recency"] = st.selectbox(
                        "📅 Recency Column",
                        csv_headers
                    )

                with c2:
                    column_map["frequency"] = st.selectbox(
                        "🔄 Frequency Column",
                        csv_headers
                    )

                with c3:
                    column_map["monetary"] = st.selectbox(
                        "💰 Monetary Column",
                        csv_headers
                    )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    column_map["customer_id"] = st.selectbox(
                        "👤 Customer ID",
                        csv_headers
                    )

                    column_map["order_date"] = st.selectbox(
                        "📅 Order Date",
                        csv_headers
                    )

                with c2:

                    column_map["order_id"] = st.selectbox(
                        "🧾 Order ID",
                        csv_headers
                    )

                    column_map["spend"] = st.selectbox(
                        "₹ Spend / Price",
                        csv_headers
                    )

            st.markdown("""
            <div style="
                background:#ecfdf5;
                border:1px solid #22c55e;
                padding:18px;
                border-radius:12px;
                margin-top:20px;
                margin-bottom:20px;
            ">
                <h4 style="margin:0;color:#166534;">
                    ✅ Data structure detected
                </h4>
                <p style="margin-top:5px;color:#166534;">
                    Your CSV is ready for MARKETRON's
                    segmentation pipeline.
                </p>
            </div>
            """, unsafe_allow_html=True)

            process_btn = st.button(
                " Process Batch Data",
                use_container_width=True
            )

            if process_btn:

                with st.spinner(
                    "Classifying customer segments..."
                ):

                    try:

                        mapped_df = process_mapped_data(
                            raw_df,
                            mapping_mode,
                            column_map
                        )

                        results_df = batch_predict_csv(
                            mapped_df
                        )

                        kpis = get_dashboard_kpis(
                            results_df
                        )

                        st.session_state.batch_results = results_df
                        st.session_state.batch_kpis = kpis
                        st.session_state.batch_processed = True

                        st.rerun()

                    except CSVProcessorError as e:
                        st.error(
                            f"⚠️ Data Ingestion Error: {e}"
                        )

                    except ValueError as e:
                        st.error(
                            f"⚠️ Data Validation Error: {e}"
                        )

                    except Exception as e:
                        st.error(
                            f"⚠️ Unexpected Error: {e}"
                        )

        except CSVProcessorError as e:
            st.error(
                f"⚠️ File Load Error: {e}"
            )

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Remove full width uploader */
[data-testid="stFileUploader"]{
    width:280px !important;
}

/* Upload button */
[data-testid="stFileUploader"] section button{
    width:140px !important;
    height:42px !important;
    border-radius:12px !important;
    font-size:16px !important;
    font-weight:600 !important;
}

/* Drop area */
[data-testid="stFileUploaderDropzone"]{
    min-height:90px !important;
    width:280px !important;
    border-radius:12px !important;
}

</style>
""", unsafe_allow_html=True)

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
        
          # ============================================================
              # 2. FULL-WIDTH VISUALIZATIONS
# ============================================================

# ------------------------------------------------------------
# 2A. SEGMENT DISTRIBUTION - CENTERED DONUT
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center; margin-top:10px;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            Segment Distribution
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Percentage of your total customer base.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

        segment_counts = (
    results_df["Segment_Name"]
    .value_counts()
    .rename_axis("Segment")
    .reset_index(name="Count")
)

        fig_pie = px.pie(
    segment_counts,
    values="Count",
    names="Segment",
    hole=0.55,
    color="Segment"
)

        fig_pie.update_traces(
    textinfo="percent",
    textposition="inside",
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Customers: %{value:,}<br>"
        "Percentage: %{percent}<extra></extra>"
    )
)

        fig_pie.update_layout(
    height=520,
    
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=0.72,
        font=dict(size=14)
    ),
    margin=dict(l=50,r=180,t=60,b=40)

)

        st.plotly_chart(
    fig_pie,
    use_container_width=True
)

        st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 2B. CUSTOMER COUNT PER SEGMENT
#     FULL-WIDTH VERTICAL BAR CHART
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            Customer Count per Segment
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Total volume of customers in each cohort.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

        fig_bar_count = px.bar(
    segment_counts,
    x="Segment",
    y="Count",
    color="Segment",
    text="Count"
)

        fig_bar_count.update_traces(
    texttemplate="%{text:,}",
    textposition="outside",
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Customers: %{y:,}<extra></extra>"
    )
)

        fig_bar_count.update_layout(
    height=500,
    margin=dict(
        t=30,
        b=80,
        l=70,
        r=40
    ),
    xaxis_title="",
    yaxis_title="Number of Customers",
    showlegend=False
)

        st.plotly_chart(
    fig_bar_count,
    use_container_width=True
)

        st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 2C. AVERAGE SPEND PER SEGMENT
#     FULL-WIDTH VERTICAL BAR CHART
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            Average Spend per Segment
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Monetary value generated by each cohort.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

        avg_spend_df = (
    results_df
    .groupby("Segment_Name", as_index=False)
    .agg(
         AverageSpend=("Monetary","mean")
         )
    
)

        fig_bar_spend = px.bar(
    avg_spend_df,
    x="Segment_Name",
    y="AverageSpend",
    color="Segment_Name",
    text="AverageSpend"
)

        fig_bar_spend.update_traces(
    texttemplate="₹%{text:,.0f}",
    textposition="outside",
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Average Spend: ₹%{y:,.2f}<extra></extra>"
    )
)

        fig_bar_spend.update_layout(
    height=500,
    margin=dict(
        t=30,
        b=80,
        l=80,
        r=40
    ),
    xaxis_title="",
    yaxis_title="Avg Spend (₹)",
    showlegend=False
)

        st.plotly_chart(
    fig_bar_spend,
    use_container_width=True
)

        st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 2D. 3D CUSTOMER UNIVERSE
#     FULL WIDTH
# ------------------------------------------------------------
        st.markdown(
    """
    <div style="text-align:center;">
        <h3 style="color:#38BDF8; margin-bottom:4px;">
            3D Customer Universe
        </h3>
        <p style="color:#94A3B8; margin-top:0;">
            Interactive map of your customer base (sampled for speed).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sample customers to keep the 3D visualization responsive
        sample_df = results_df.sample(
    min(1000, len(results_df)),
    random_state=42
)

        fig_3d = px.scatter_3d(
    sample_df,
    x="Recency",
    y="Frequency",
    z="Monetary",
    color="Segment_Name",
    opacity=0.7,
    size_max=10
)

        fig_3d.update_layout(
    height=650,
    margin=dict(
        t=10,
        b=10,
        l=10,
        r=10
    )
)

        st.plotly_chart(
    fig_3d,
    use_container_width=True
)
        # 4. FULL INTERACTIVE DATABASE
        st.markdown("### 📋 Customer Database")
        st.write("Your original data, now enhanced with ML segment predictions. You can sort and filter this table directly.")
        st.dataframe(results_df, use_container_width=True, height=250)
        
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
        generate_btn = st.button(" Generate Approach", key="batch_generate_btn", width='stretch')
        
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
        with st.status(f"🧠 Processing Strategy for {target_segment}...", expanded=True) as status:
            st.write("Connecting to Marketron AI...")
            time.sleep(0.4)
            st.write("Crafting your AI-powered campaign approach...")
            
            initial_prompt = "Provide a brief, high-level overview of exactly how we should market to this specific segment."
            st.session_state.batch_chat_history = generate_action_response(
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

        if st.session_state.get("batch_scroll_pending", False):
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