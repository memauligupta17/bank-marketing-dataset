"""
Bank Marketing — Term Deposit Subscription Predictor
======================================================
A Streamlit app for the Bank Marketing ML project.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    JOB_OPTIONS, MARITAL_OPTIONS, EDUCATION_OPTIONS, YES_NO_OPTIONS,
    CONTACT_OPTIONS, MONTH_OPTIONS, POUTCOME_OPTIONS,
    train_model, build_input_row,
)

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing | Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme / CSS — navy & gold "banking" palette
# ----------------------------------------------------------------------------
PRIMARY_NAVY = "#0B2545"
DEEP_NAVY = "#081C36"
ACCENT_GOLD = "#D4AF37"
SOFT_GOLD = "#F4E4BC"
SUCCESS_GREEN = "#1E7145"
SUCCESS_GREEN_LIGHT = "#E6F4EA"
DANGER_RED = "#A4262C"
DANGER_RED_LIGHT = "#FDECEA"
CARD_BG = "#FFFFFF"
PAGE_BG = "#F4F6F9"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {PAGE_BG};
    }}

    /* ---- Top banner ---- */
    .bm-header {{
        background: linear-gradient(135deg, {PRIMARY_NAVY} 0%, {DEEP_NAVY} 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        box-shadow: 0 6px 20px rgba(8, 28, 54, 0.25);
        border: 1px solid rgba(212, 175, 55, 0.35);
    }}
    .bm-header h1 {{
        color: #FFFFFF;
        font-size: 2.1rem;
        margin: 0 0 0.35rem 0;
        font-weight: 800;
        letter-spacing: 0.2px;
    }}
    .bm-header p {{
        color: {SOFT_GOLD};
        font-size: 1.02rem;
        margin: 0;
        opacity: 0.92;
    }}
    .bm-badge {{
        display: inline-block;
        background: rgba(212,175,55,0.18);
        color: {ACCENT_GOLD};
        border: 1px solid rgba(212,175,55,0.5);
        border-radius: 999px;
        padding: 0.15rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        margin-top: 0.7rem;
    }}

    /* ---- Section / card containers ---- */
    .bm-card {{
        background: {CARD_BG};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #E4E8EF;
        box-shadow: 0 2px 10px rgba(16, 30, 54, 0.05);
        margin-bottom: 1.2rem;
    }}
    .bm-card h3 {{
        margin-top: 0;
        color: {PRIMARY_NAVY};
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY_NAVY} 0%, {DEEP_NAVY} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #EEF2F8 !important;
    }}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stRadio label {{
        font-weight: 600 !important;
        color: {SOFT_GOLD} !important;
    }}
    section[data-testid="stSidebar"] .stExpander {{
        border: 1px solid rgba(212,175,55,0.28) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.03) !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(212,175,55,0.25);
    }}

    /* ---- Prediction result boxes ---- */
    .bm-result-box {{
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin-bottom: 1.2rem;
        border: 2px solid;
    }}
    .bm-result-yes {{
        background: linear-gradient(135deg, {SUCCESS_GREEN_LIGHT} 0%, #d3f0dc 100%);
        border-color: {SUCCESS_GREEN};
    }}
    .bm-result-no {{
        background: linear-gradient(135deg, {DANGER_RED_LIGHT} 0%, #fbdcda 100%);
        border-color: {DANGER_RED};
    }}
    .bm-result-title {{
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0.2rem 0 0.15rem 0;
    }}
    .bm-result-yes .bm-result-title {{ color: {SUCCESS_GREEN}; }}
    .bm-result-no .bm-result-title {{ color: {DANGER_RED}; }}
    .bm-result-sub {{
        font-size: 1rem;
        color: #33414f;
        margin: 0;
    }}
    .bm-result-icon {{ font-size: 2.6rem; margin-bottom: 0.2rem; }}

    /* ---- Metric pills ---- */
    .bm-metric {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
        border: 1px solid #E4E8EF;
    }}
    .bm-metric .val {{ font-size: 1.4rem; font-weight: 800; color: {PRIMARY_NAVY}; }}
    .bm-metric .lbl {{ font-size: 0.8rem; color: #6b7686; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }}

    div.stButton > button {{
        background: linear-gradient(135deg, {ACCENT_GOLD} 0%, #b8912a 100%);
        color: {DEEP_NAVY};
        font-weight: 800;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        width: 100%;
        font-size: 1.05rem;
        transition: transform 0.05s ease-in-out;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(212,175,55,0.35);
        color: {DEEP_NAVY};
    }}

    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model (cached — trained once per session/server)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the Gradient Boosting model...")
def get_model():
    model, feature_columns, metrics = train_model()
    return model, feature_columns, metrics


model, feature_columns, metrics = get_model()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="bm-header">
    <h1>🏦 Bank Marketing — Term Deposit Predictor</h1>
    <p>Predict whether a customer is likely to subscribe to a term deposit based on
    their profile and campaign interaction history.</p>
    <span class="bm-badge">GRADIENT BOOSTING CLASSIFIER</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar — grouped, user-friendly inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧾 Customer Input Form")
    st.caption("Fill in the customer & campaign details, then hit **Predict**.")
    st.markdown("---")

    with st.expander("👤 Customer Profile", expanded=True):
        age = st.slider("Age", 18, 95, 35)
        job = st.selectbox("Job", JOB_OPTIONS, index=JOB_OPTIONS.index("management"))
        marital = st.selectbox("Marital Status", MARITAL_OPTIONS, index=0)
        education = st.selectbox("Education", EDUCATION_OPTIONS, index=1)

    with st.expander("💰 Financial Details", expanded=True):
        balance = st.number_input("Account Balance (€)", -8000, 100000, 1500, step=100)
        default = st.radio("Has Credit in Default?", YES_NO_OPTIONS, index=1, horizontal=True)
        housing = st.radio("Has Housing Loan?", YES_NO_OPTIONS, index=1, horizontal=True)
        loan = st.radio("Has Personal Loan?", YES_NO_OPTIONS, index=1, horizontal=True)

    with st.expander("📞 Current Campaign Contact", expanded=True):
        contact = st.selectbox("Contact Communication Type", CONTACT_OPTIONS, index=0)
        month = st.selectbox("Last Contact Month", MONTH_OPTIONS, index=4)
        day = st.slider("Last Contact Day of Month", 1, 31, 15)
        duration = st.slider("Last Contact Duration (seconds)", 0, 3800, 300, step=10)
        campaign = st.slider("Number of Contacts (this campaign)", 1, 40, 2)

    with st.expander("📊 Previous Campaign History", expanded=False):
        previous = st.slider("Number of Contacts Before This Campaign", 0, 20, 0)
        pdays = st.slider(
            "Days Since Last Contact (-1 = never contacted)", -1, 900, -1
        )
        poutcome = st.selectbox("Previous Campaign Outcome", POUTCOME_OPTIONS, index=3)

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Subscription", use_container_width=True)

# ----------------------------------------------------------------------------
# Main layout — Tabs
# ----------------------------------------------------------------------------
tab_predict, tab_insights, tab_about = st.tabs(
    ["🔮 Prediction", "📈 Model Insights", "ℹ️ About"]
)

# ============================== PREDICTION TAB ==============================
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    raw_input = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome,
    }

    with left:
        st.markdown('<div class="bm-card">', unsafe_allow_html=True)
        st.markdown("### 🧍 Customer Summary")
        s1, s2 = st.columns(2)
        with s1:
            st.write(f"**Age:** {age}")
            st.write(f"**Job:** {job.title()}")
            st.write(f"**Marital:** {marital.title()}")
            st.write(f"**Education:** {education.title()}")
            st.write(f"**Balance:** €{balance:,}")
            st.write(f"**Default:** {default.title()}")
            st.write(f"**Housing Loan:** {housing.title()}")
            st.write(f"**Personal Loan:** {loan.title()}")
        with s2:
            st.write(f"**Contact Type:** {contact.title()}")
            st.write(f"**Month:** {month.upper()}")
            st.write(f"**Day:** {day}")
            st.write(f"**Call Duration:** {duration}s")
            st.write(f"**Campaign Contacts:** {campaign}")
            st.write(f"**Previous Contacts:** {previous}")
            st.write(f"**Days Since Last Contact:** {pdays}")
            st.write(f"**Previous Outcome:** {poutcome.title()}")
        st.markdown('</div>', unsafe_allow_html=True)

        if not predict_clicked:
            st.info("👈 Adjust the customer details in the sidebar and click "
                     "**Predict Subscription** to see the result.")

    with right:
        if predict_clicked:
            input_row = build_input_row(raw_input, feature_columns)
            proba_yes = float(model.predict_proba(input_row)[0, 1])
            proba_no = 1 - proba_yes
            will_subscribe = proba_yes >= 0.5

            if will_subscribe:
                st.markdown(f"""
                <div class="bm-result-box bm-result-yes">
                    <div class="bm-result-icon">✅</div>
                    <p class="bm-result-title">Likely to SUBSCRIBE</p>
                    <p class="bm-result-sub">Confidence: {proba_yes*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("This customer looks like a strong candidate for the "
                            "term deposit offer. Recommend prioritizing contact.")
            else:
                st.markdown(f"""
                <div class="bm-result-box bm-result-no">
                    <div class="bm-result-icon">❌</div>
                    <p class="bm-result-title">Likely to DECLINE</p>
                    <p class="bm-result-sub">Confidence: {proba_no*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.warning("This customer is less likely to subscribe. Consider "
                           "a lighter-touch or alternative offer to avoid over-contacting.")

            # Probability gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_yes * 100,
                number={"suffix": "%", "font": {"size": 34, "color": PRIMARY_NAVY}},
                title={"text": "Subscription Probability", "font": {"size": 15, "color": "#4a5568"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8a97a8"},
                    "bar": {"color": ACCENT_GOLD, "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": DANGER_RED_LIGHT},
                        {"range": [40, 60], "color": "#FFF3D6"},
                        {"range": [60, 100], "color": SUCCESS_GREEN_LIGHT},
                    ],
                    "threshold": {
                        "line": {"color": PRIMARY_NAVY, "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            gauge.update_layout(height=260, margin=dict(l=25, r=25, t=50, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, use_container_width=True)

            with st.expander("🔍 What influenced this prediction?"):
                top_feats = metrics["feature_importance"].head(6)
                st.caption("Top overall drivers of the model (from training):")
                st.bar_chart(top_feats)
        else:
            st.markdown('<div class="bm-card" style="text-align:center; padding-top:3rem; padding-bottom:3rem;">',
                        unsafe_allow_html=True)
            st.markdown("### ⏳ Awaiting Input")
            st.write("Your prediction result, confidence gauge, and key drivers "
                     "will appear here once you click **Predict Subscription**.")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================== INSIGHTS TAB ==============================
with tab_insights:
    st.markdown("### 📈 Model Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (label, val) in zip(
        [m1, m2, m3, m4, m5],
        [
            ("Accuracy", metrics["accuracy"]),
            ("Precision", metrics["precision"]),
            ("Recall", metrics["recall"]),
            ("F1 Score", metrics["f1"]),
            ("ROC AUC", metrics["roc_auc"]),
        ],
    ):
        with col:
            st.markdown(
                f'<div class="bm-metric"><div class="val">{val*100:.1f}%</div>'
                f'<div class="lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="bm-card">', unsafe_allow_html=True)
        st.markdown("#### 🔑 Top Feature Importances")
        top_n = metrics["feature_importance"].head(12).sort_values()
        fig = go.Figure(go.Bar(
            x=top_n.values, y=top_n.index, orientation="h",
            marker_color=ACCENT_GOLD,
        ))
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Importance",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="bm-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 ROC Curve")
        fpr, tpr = metrics["roc_curve"]
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      line=dict(color=PRIMARY_NAVY, width=3),
                                      name=f"ROC (AUC={metrics['roc_auc']:.3f})"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(color="#c8ccd4", dash="dash"),
                                      name="Random"))
        roc_fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(x=0.4, y=0.08),
        )
        st.plotly_chart(roc_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="bm-card">', unsafe_allow_html=True)
    st.markdown("#### 🧮 Confusion Matrix (Test Set)")
    cm = metrics["confusion_matrix"]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
        colorscale=[[0, "#EAF1FB"], [1, PRIMARY_NAVY]],
        text=cm, texttemplate="%{text}", textfont={"size": 18},
        showscale=False,
    ))
    cm_fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================== ABOUT TAB ==============================
with tab_about:
    st.markdown('<div class="bm-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About This Project")
    st.write(
        "This app predicts whether a bank customer will **subscribe to a term "
        "deposit**, based on the UCI *Bank Marketing* dataset structure "
        "(demographic, financial, and campaign-contact attributes)."
    )
    st.markdown("""
**Pipeline (matches the project notebook):**
- One-hot encoding (`drop_first=True`) for categorical fields: job, marital,
  education, default, housing, loan, contact, month, poutcome
- Numeric features used as-is: age, balance, day, duration, campaign, pdays, previous
- Models compared: Logistic Regression, Random Forest, **Gradient Boosting (final choice)**
- Final model selected for the best balance of accuracy, precision, recall, and F1

**Key business drivers identified:** call `duration`, previous campaign
`outcome`, number of `campaign` contacts, `age`, and account `balance`.
    """)
    st.info(
        "📌 **Note:** The original notebook's raw dataset file (`bank (1).csv`) "
        "wasn't packaged with this project export, so this app trains its "
        "Gradient Boosting model on a realistic synthetic dataset built with "
        "matching columns, categories, and directional relationships from the "
        "notebook's EDA. Drop your real `bank.csv` into a `data/` folder "
        "(see `model_utils.py`) to train on the authentic dataset instead."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f"<div style='text-align:center; color:#8a97a8; padding:1rem 0; font-size:0.85rem;'>"
    f"Bank Marketing Term Deposit Predictor &nbsp;•&nbsp; Built with Streamlit</div>",
    unsafe_allow_html=True,
)
