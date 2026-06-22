import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Loan Simulator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0f0;
}

/* Header */
.hero-header {
    background: linear-gradient(90deg, #6c63ff 0%, #a855f7 60%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: #9d9dc7;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    text-align: center;
}

/* Result boxes */
.result-approved {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 2px solid #10b981;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}

.result-rejected {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border: 2px solid #ef4444;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
}

.result-title-approved {
    font-size: 2rem;
    font-weight: 800;
    color: #34d399;
}

.result-title-rejected {
    font-size: 2rem;
    font-weight: 800;
    color: #f87171;
}

/* Step pills */
.step-pill {
    background: rgba(108,99,255,0.2);
    border: 1px solid rgba(108,99,255,0.5);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    color: #c4b5fd;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9d9dc7;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #6c63ff, #a855f7) !important;
    color: white !important;
}

/* Inputs */
.stNumberInput input, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #6c63ff 0%, #a855f7 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #c4b5fd;
    margin: 1.2rem 0 0.6rem;
    border-left: 3px solid #6c63ff;
    padding-left: 0.75rem;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training decision tree model…")
def load_model():
    sys.path.insert(0, os.path.dirname(__file__))
    from backend.model import LoanSimulatorModel
    data_path = os.path.join(os.path.dirname(__file__), "data", "loan_approval_dataset.csv")
    sim = LoanSimulatorModel(data_path)
    info = sim.train()
    return sim, info

simulator, train_info = load_model()

# ── Load dataset (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    path = os.path.join(os.path.dirname(__file__), "data", "loan_approval_dataset.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Simulator")
    st.markdown("---")
    acc = train_info.get("accuracy", 0)
    st.metric("Model Accuracy", f"{acc*100:.1f}%")
    st.metric("Algorithm", "Decision Tree")
    st.metric("Max Depth", "4 levels")
    st.markdown("---")
    total = len(df)
    approved = (df["loan_status"] == "Approved").sum()
    rejected = total - approved
    st.metric("Dataset Size", f"{total:,} records")
    st.metric("Approved", f"{approved:,} ({approved/total*100:.1f}%)")
    st.metric("Rejected", f"{rejected:,} ({rejected/total*100:.1f}%)")
    st.markdown("---")
    st.caption("Explainable AI · Decision Tree · Loan Prediction")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-header">🏦 Explainable Loan Rejection Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered loan decisions with transparent, step-by-step reasoning.</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Loan Application", "🌳 Explainability & Rules", "📊 Dataset Explorer"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Loan Application Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Fill in the Applicant Details")
    st.markdown("Adjust the fields below and click **Check Approval Status** to get an instant, explainable decision.")

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">💰 Financial Profile</div>', unsafe_allow_html=True)
            income_annum = st.number_input("Annual Income (₹)", min_value=0, max_value=100_000_000, value=500_000, step=10_000)
            loan_amount  = st.number_input("Requested Loan Amount (₹)", min_value=0, max_value=200_000_000, value=1_000_000, step=10_000)
            loan_term    = st.number_input("Loan Term (Months)", min_value=1, max_value=360, value=12)
            cibil_score  = st.slider("CIBIL Score", min_value=300, max_value=900, value=700,
                                     help="Score above 550 generally improves approval odds.")

        with col2:
            st.markdown('<div class="section-header">👤 Personal Details</div>', unsafe_allow_html=True)
            education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            dependents    = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

            st.markdown('<div class="section-header">🏠 Asset Details</div>', unsafe_allow_html=True)
            bank_asset        = st.number_input("Bank Asset Value (₹)",         min_value=0, value=0, step=10_000)
            residential_asset = st.number_input("Residential Assets Value (₹)", min_value=0, value=0, step=10_000)
            commercial_asset  = st.number_input("Commercial Assets Value (₹)",  min_value=0, value=0, step=10_000)
            luxury_asset      = st.number_input("Luxury Assets Value (₹)",      min_value=0, value=0, step=10_000)

        submitted = st.form_submit_button("🔍 Check Approval Status", use_container_width=True)

    # ── Result ─────────────────────────────────────────────────────────────────
    if submitted:
        form_data = {
            "income_annum":             income_annum,
            "loan_amount":              loan_amount,
            "loan_term":                loan_term,
            "cibil_score":              cibil_score,
            "education":                f" {education}",
            "self_employed":            f" {self_employed}",
            "no_of_dependents":         dependents,
            "bank_asset_value":         bank_asset,
            "residential_assets_value": residential_asset,
            "commercial_assets_value":  commercial_asset,
            "luxury_assets_value":      luxury_asset,
        }

        with st.spinner("Evaluating your application…"):
            result = simulator.predict_explainable(form_data)

        st.markdown("---")
        approved_flag = result["status"] == "Approved"

        if approved_flag:
            st.markdown("""
            <div class="result-approved">
                <div class="result-title-approved">✅ Application Approved!</div>
                <p style="color:#a7f3d0;margin-top:0.5rem;">Congratulations! Your application meets the criteria.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-rejected">
                <div class="result-title-rejected">❌ Application Rejected</div>
                <p style="color:#fca5a5;margin-top:0.5rem;">Your application did not meet the required criteria.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🧠 Decision Path Explanation")
        st.caption("The model evaluated the following rules step-by-step to reach its decision:")

        for i, step in enumerate(result["decision_path"], 1):
            icon = "✅" if approved_flag else "⚠️"
            st.markdown(f"""
            <div class="step-pill">
                <span style="background:rgba(108,99,255,0.4);padding:0.2rem 0.6rem;border-radius:6px;font-weight:700;font-size:0.85rem;">Step {i}</span>
                <span>{icon} {step}</span>
            </div>""", unsafe_allow_html=True)

        # Quick summary metrics
        st.markdown("#### 📊 Key Metrics at a Glance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CIBIL Score",    cibil_score, delta="Good" if cibil_score > 700 else ("Fair" if cibil_score > 550 else "Poor"))
        m2.metric("Loan Amount",    f"₹{loan_amount:,}")
        m3.metric("Annual Income",  f"₹{income_annum:,}")
        total_assets = bank_asset + residential_asset + commercial_asset + luxury_asset
        m4.metric("Total Assets",   f"₹{total_assets:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Explainability & Rules
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🌳 Global Decision Tree Rules")
    st.caption("These are the exact rules learned from the training dataset. The tree has a max depth of 4 to keep it human-readable.")

    with st.expander("📜 View Full Decision Tree Rules", expanded=True):
        st.code(train_info.get("rules", "Rules not available."), language="text")

    st.markdown("---")
    st.markdown("### 📈 Feature Importances")
    st.caption("Which factors matter most for the loan decision?")

    feature_names = simulator.feature_names
    importances   = simulator.model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    feat_df["Importance %"] = (feat_df["Importance"] * 100).round(2)

    st.dataframe(
        feat_df[["Feature", "Importance %"]].style.background_gradient(cmap="Purples", subset=["Importance %"]),
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(feat_df.set_index("Feature")["Importance %"])

    st.markdown("---")
    st.markdown("### ℹ️ How the Model Works")
    st.info("""
**Algorithm:** Decision Tree Classifier (scikit-learn)

**Training Data:** Real-world Indian loan approval dataset with features including CIBIL score,
annual income, loan amount & term, assets, education, and employment status.

**Explainability:** Instead of a black-box output, the model exposes the exact sequence of 
IF-THEN rules it followed to reach the decision — making every prediction fully transparent.

**Max Depth = 4** ensures the rules stay short enough for a human to understand.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Dataset Explorer")
    st.caption("Explore the underlying loan approval dataset used to train the model.")

    # Summary stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records",   f"{len(df):,}")
    s2.metric("Approved",        f"{(df['loan_status']=='Approved').sum():,}")
    s3.metric("Rejected",        f"{(df['loan_status']=='Rejected').sum():,}")
    s4.metric("Approval Rate",   f"{(df['loan_status']=='Approved').mean()*100:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### CIBIL Score Distribution")
        cibil_data = df.groupby("loan_status")["cibil_score"].apply(list)
        chart_data = pd.DataFrame({
            "Approved": pd.Series(cibil_data.get("Approved", [])),
            "Rejected": pd.Series(cibil_data.get("Rejected", [])),
        })
        st.line_chart(
            df.groupby(["cibil_score", "loan_status"]).size().unstack(fill_value=0),
            use_container_width=True,
        )

    with col_b:
        st.markdown("#### Loan Amount Distribution (₹)")
        amount_data = df.groupby("loan_status")["loan_amount"].apply(list)
        bins = pd.cut(df["loan_amount"], bins=20)
        loan_hist = df.groupby([bins, "loan_status"]).size().unstack(fill_value=0)
        loan_hist.index = loan_hist.index.astype(str)
        st.bar_chart(loan_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Income vs. Loan Amount by Status")
    scatter_df = df[["income_annum", "loan_amount", "loan_status"]].copy()
    scatter_df = scatter_df.rename(columns={"income_annum": "Annual Income", "loan_amount": "Loan Amount"})
    # Show scatter via a simple pivot table
    approved_df = scatter_df[scatter_df["loan_status"] == "Approved"][["Annual Income", "Loan Amount"]].head(300)
    rejected_df = scatter_df[scatter_df["loan_status"] == "Rejected"][["Annual Income", "Loan Amount"]].head(300)
    scatter_combined = pd.concat([
        approved_df.assign(Status="Approved"),
        rejected_df.assign(Status="Rejected"),
    ])
    st.dataframe(scatter_combined.head(50), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Raw Dataset Sample")
    show_n = st.slider("Rows to display", 5, 100, 20)
    st.dataframe(df.head(show_n), use_container_width=True, hide_index=True)
