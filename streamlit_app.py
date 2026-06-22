import streamlit as st
import pandas as pd
import numpy as np
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Loan Simulator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

.hero-header {
    background: linear-gradient(90deg, #6c63ff 0%, #a855f7 60%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.hero-sub { color: #9d9dc7; font-size: 1rem; margin-bottom: 1rem; }

.result-approved {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #10b981;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-rejected {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #ef4444;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1rem;
}
.approved-title { font-size: 2rem; font-weight: 800; color: #34d399; }
.rejected-title { font-size: 2rem; font-weight: 800; color: #f87171; }

.step-pill {
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.45rem;
    font-size: 0.95rem;
    color: #c4b5fd;
}
.step-num {
    background: rgba(108,99,255,0.4);
    padding: 0.15rem 0.55rem;
    border-radius: 5px;
    font-weight: 700;
    font-size: 0.82rem;
    margin-right: 0.6rem;
}
.sec-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c4b5fd;
    border-left: 3px solid #6c63ff;
    padding-left: 0.7rem;
    margin: 1.1rem 0 0.5rem;
}

section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#6c63ff,#a855f7) !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(90deg,#6c63ff,#a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
}
.stButton > button:hover { opacity: 0.87; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Inline Model (no backend import needed) ────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

class LoanModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False

    def train(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip()
        if "loan_id" in df.columns:
            df = df.drop("loan_id", axis=1)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        X = df.drop("loan_status", axis=1)
        y = df["loan_status"].apply(lambda x: 1 if x == "Approved" else 0)
        self.feature_names = X.columns.tolist()

        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        self.model.fit(X, y)
        self.is_trained = True
        acc = self.model.score(X, y)
        rules = export_text(self.model, feature_names=self.feature_names)
        return acc, rules

    def predict(self, input_data):
        inp = pd.DataFrame([input_data])
        for col in self.feature_names:
            if col not in inp.columns:
                inp[col] = 0
            if col in self.label_encoders:
                val = str(inp[col].iloc[0]).strip()
                le = self.label_encoders[col]
                inp[col] = le.transform([val])[0] if val in le.classes_ else 0
        X = inp[self.feature_names]
        pred = self.model.predict(X)[0]
        status = "Approved" if pred == 1 else "Rejected"

        node_ind = self.model.decision_path(X)
        leaf_id  = self.model.apply(X)[0]
        feat     = self.model.tree_.feature
        thresh   = self.model.tree_.threshold
        nodes    = node_ind.indices[node_ind.indptr[0]:node_ind.indptr[1]]

        reasons = []
        for nid in nodes:
            if nid == leaf_id:
                continue
            fname = self.feature_names[feat[nid]]
            fval  = X.iloc[0, feat[nid]]
            tval  = thresh[nid]
            disp  = fval
            if fname in self.label_encoders:
                cls = self.label_encoders[fname].classes_
                if int(fval) < len(cls):
                    disp = cls[int(fval)]
            if fval <= tval:
                reasons.append(f"{fname} ({disp}) ≤ {tval:.1f}")
            else:
                reasons.append(f"{fname} ({disp}) > {tval:.1f}")
        return status, reasons

# ── Cache: load data & train ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training decision tree model…")
def get_model_and_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "loan_approval_dataset.csv")
    df = pd.read_csv(path)
    # Strip column names and all string values so downstream code works cleanly
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    if "loan_id" in df.columns:
        df = df.drop("loan_id", axis=1)
    model = LoanModel()
    acc, rules = model.train(df)
    return model, df, acc, rules

model, df, accuracy, tree_rules = get_model_and_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Simulator")
    st.markdown("---")
    st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
    st.metric("Algorithm", "Decision Tree")
    st.metric("Max Depth",  "4 levels")
    st.markdown("---")
    total    = len(df)
    approved = (df["loan_status"].str.strip() == "Approved").sum()
    rejected = total - approved
    st.metric("Dataset Size", f"{total:,}")
    st.metric("Approved",     f"{approved:,}  ({approved/total*100:.1f}%)")
    st.metric("Rejected",     f"{rejected:,}  ({rejected/total*100:.1f}%)")
    st.markdown("---")
    st.caption("Explainable AI · Decision Tree · Loan Prediction")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-header">🏦 Explainable Loan Rejection Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered loan decisions with transparent, step-by-step reasoning.</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋  Loan Application", "🌳  Explainability & Rules", "📊  Dataset Explorer"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Fill in the Applicant Details")

    with st.form("loan_form"):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-header">💰 Financial Profile</div>', unsafe_allow_html=True)
            income       = st.number_input("Annual Income (₹)",        min_value=0, max_value=100_000_000, value=500_000,   step=10_000)
            loan_amount  = st.number_input("Loan Amount (₹)",          min_value=0, max_value=200_000_000, value=1_000_000, step=10_000)
            loan_term    = st.number_input("Loan Term (months)",        min_value=1, max_value=360,         value=12)
            cibil        = st.slider(      "CIBIL Score",               300, 900, 700,
                                           help="Scores above 550 strongly improve approval odds.")

        with c2:
            st.markdown('<div class="sec-header">👤 Personal Details</div>', unsafe_allow_html=True)
            education     = st.selectbox("Education",     ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            dependents    = st.number_input("Dependents", min_value=0, max_value=20, value=0)
            st.markdown('<div class="sec-header">🏠 Assets</div>', unsafe_allow_html=True)
            bank_asset    = st.number_input("Bank Asset Value (₹)",        min_value=0, value=0, step=10_000)
            res_asset     = st.number_input("Residential Asset Value (₹)", min_value=0, value=0, step=10_000)
            com_asset     = st.number_input("Commercial Asset Value (₹)",  min_value=0, value=0, step=10_000)
            lux_asset     = st.number_input("Luxury Asset Value (₹)",      min_value=0, value=0, step=10_000)

        submitted = st.form_submit_button("🔍  Check Approval Status", use_container_width=True)

    if submitted:
        form_data = {
            "income_annum":             income,
            "loan_amount":              loan_amount,
            "loan_term":                loan_term,
            "cibil_score":              cibil,
            "education":                education,
            "self_employed":            self_employed,
            "no_of_dependents":         dependents,
            "bank_asset_value":         bank_asset,
            "residential_assets_value": res_asset,
            "commercial_assets_value":  com_asset,
            "luxury_assets_value":      lux_asset,
        }

        with st.spinner("Evaluating application…"):
            status, reasons = model.predict(form_data)

        st.markdown("---")
        is_approved = status == "Approved"

        if is_approved:
            st.markdown(
                '<div class="result-approved">'
                '<div class="approved-title">✅ Application Approved!</div>'
                '<p style="color:#a7f3d0;margin-top:0.4rem">Congratulations! Your profile meets the lending criteria.</p>'
                '</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-rejected">'
                '<div class="rejected-title">❌ Application Rejected</div>'
                '<p style="color:#fca5a5;margin-top:0.4rem">Your profile did not meet the required criteria.</p>'
                '</div>', unsafe_allow_html=True)

        st.markdown("#### 🧠 Decision Path — Step by Step")
        for i, step in enumerate(reasons, 1):
            icon = "✅" if is_approved else "⚠️"
            st.markdown(
                f'<div class="step-pill"><span class="step-num">Step {i}</span>{icon} {step}</div>',
                unsafe_allow_html=True)

        st.markdown("#### 📊 Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CIBIL Score",   cibil,          delta="Good" if cibil>700 else ("Fair" if cibil>550 else "Poor"))
        m2.metric("Loan Amount",   f"₹{loan_amount:,}")
        m3.metric("Annual Income", f"₹{income:,}")
        m4.metric("Total Assets",  f"₹{bank_asset+res_asset+com_asset+lux_asset:,}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🌳 Global Decision Tree Rules")
    with st.expander("📜 Full Decision Tree (max depth = 4)", expanded=True):
        st.code(tree_rules, language="text")

    st.markdown("---")
    st.markdown("### 📈 Feature Importances")
    imps = model.model.feature_importances_
    feat_df = (
        pd.DataFrame({"Feature": model.feature_names, "Importance %": (imps * 100).round(2)})
        .sort_values("Importance %", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.bar_chart(feat_df.set_index("Feature")["Importance %"])

    st.markdown("---")
    st.info("""
**Algorithm:** Decision Tree Classifier — scikit-learn, max_depth=4

**Why explainable?** Every prediction follows a transparent IF-THEN rule path readable by any human.

**Dataset:** Real-world Indian loan approval records with 4,269 applications.
    """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Dataset Explorer")

    s1, s2, s3, s4 = st.columns(4)
    total_r   = len(df)
    appr_r    = (df["loan_status"].str.strip() == "Approved").sum()
    s1.metric("Total Records",  f"{total_r:,}")
    s2.metric("Approved",       f"{appr_r:,}")
    s3.metric("Rejected",       f"{total_r - appr_r:,}")
    s4.metric("Approval Rate",  f"{appr_r/total_r*100:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### CIBIL Score Distribution by Status")
        cibil_chart = (
            df.assign(loan_status=df["loan_status"].str.strip())
              .groupby(["cibil_score", "loan_status"])
              .size().unstack(fill_value=0)
        )
        st.line_chart(cibil_chart, use_container_width=True)

    with col_b:
        st.markdown("#### Loan Amount Distribution by Status")
        df2 = df.assign(loan_status=df["loan_status"].str.strip())
        bins = pd.cut(df2["loan_amount"], bins=15)
        loan_chart = df2.groupby([bins, "loan_status"]).size().unstack(fill_value=0)
        loan_chart.index = loan_chart.index.astype(str)
        st.bar_chart(loan_chart, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Raw Dataset Sample")
    n = st.slider("Rows to show", 5, 100, 20)
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)
