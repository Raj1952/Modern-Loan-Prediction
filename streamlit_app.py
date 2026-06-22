import streamlit as st
import pandas as pd
import numpy as np
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ — Explainable Loan Decisions",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Neo-Brutalist CSS (21st.dev inspired) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #F5F0E8 !important;
    color: #0A0A0A !important;
}

/* Kill default Streamlit padding */
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* ── Top nav bar ── */
.nav-bar {
    background: #0A0A0A;
    padding: 0 2.5rem;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid #0A0A0A;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #F5F0E8;
    letter-spacing: 0.05em;
}
.nav-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #888;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Hero ── */
.hero {
    border-bottom: 2px solid #0A0A0A;
    padding: 4rem 2.5rem 3rem;
    background: #F5F0E8;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #666;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 2px;
    background: #0A0A0A;
}
.hero-title {
    font-size: clamp(2.8rem, 6vw, 5.5rem);
    font-weight: 700;
    line-height: 0.95;
    letter-spacing: -0.03em;
    color: #0A0A0A;
    margin-bottom: 1.5rem;
}
.hero-title span {
    display: block;
    color: #0A0A0A;
}
.hero-title .accent {
    color: transparent;
    -webkit-text-stroke: 2px #0A0A0A;
}
.hero-desc {
    font-size: 1rem;
    color: #555;
    max-width: 480px;
    line-height: 1.6;
}

/* ── Stat strip ── */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 2px solid #0A0A0A;
    background: #0A0A0A;
}
.stat-item {
    padding: 1.5rem 2rem;
    border-right: 1px solid #222;
    color: #F5F0E8;
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    display: block;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label {
    font-size: 0.72rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Main layout ── */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    min-height: calc(100vh - 200px);
    border-bottom: 2px solid #0A0A0A;
}
.form-panel {
    border-right: 2px solid #0A0A0A;
    padding: 2.5rem;
}
.result-panel {
    padding: 2.5rem;
    background: #F5F0E8;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #999;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #D0C8B8;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '//';
    color: #CCC;
}

/* ── Field groups ── */
.field-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    border: 1.5px solid #0A0A0A;
    margin-bottom: 1rem;
    background: #0A0A0A;
}
.field-cell {
    background: #F5F0E8;
    padding: 0.8rem 1rem;
}
.field-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.25rem;
}

/* ── Streamlit input overrides — force cream bg + black text ── */

/* Number inputs: container, inner wrapper, actual input element */
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background: #F5F0E8 !important;
    background-color: #F5F0E8 !important;
    border: none !important;
    border-bottom: 1.5px solid #AAA !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
[data-testid="stNumberInput"] input {
    background: #F5F0E8 !important;
    background-color: #F5F0E8 !important;
    color: #0A0A0A !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: none !important;
    caret-color: #0A0A0A !important;
}
[data-testid="stNumberInput"] input:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Stepper +/- buttons */
[data-testid="stNumberInput"] button,
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"] {
    background: #E8E2D6 !important;
    background-color: #E8E2D6 !important;
    color: #0A0A0A !important;
    border: none !important;
    border-left: 1px solid #C8BFA8 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #0A0A0A !important;
    color: #F5F0E8 !important;
}
[data-testid="stNumberInput"] button svg path {
    stroke: #0A0A0A !important;
    fill: #0A0A0A !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="base-input"] {
    background: #F5F0E8 !important;
    background-color: #F5F0E8 !important;
    border: none !important;
    border-bottom: 1.5px solid #AAA !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: #0A0A0A !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #0A0A0A !important;
}
[data-testid="stSelectbox"] svg { color: #0A0A0A !important; }

/* Selectbox dropdown list */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: #F5F0E8 !important;
    border: 1.5px solid #0A0A0A !important;
    border-radius: 0 !important;
}
[data-baseweb="popover"] li {
    color: #0A0A0A !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    background: #F5F0E8 !important;
}
[data-baseweb="popover"] li:hover {
    background: #0A0A0A !important;
    color: #F5F0E8 !important;
}

/* Slider track */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #0A0A0A !important;
    border-radius: 0 !important;
    width: 14px !important;
    height: 14px !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #0A0A0A !important;
}


/* ── Submit button (both st.button AND st.form_submit_button) ── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-secondaryFormSubmit"],
[kind="secondaryFormSubmit"],
button[kind="secondaryFormSubmit"] {
    background: #0A0A0A !important;
    background-color: #0A0A0A !important;
    color: #F5F0E8 !important;
    border: 2px solid #0A0A0A !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    font-weight: 700 !important;
    transition: all 0.12s ease !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: #F5F0E8 !important;
    background-color: #F5F0E8 !important;
    color: #0A0A0A !important;
}

/* ── Dataframe light theme ── */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] > div {
    background: #FDFAF4 !important;
    border: 1.5px solid #0A0A0A !important;
    border-radius: 0 !important;
}

/* ── Slider label + value ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {
    color: #0A0A0A !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #888 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
}

/* ── Decision tree styled block ── */
.tree-container {
    background: #FDFAF4;
    border: 1.5px solid #0A0A0A;
    padding: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.9;
    overflow-x: auto;
}
.tree-line {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    white-space: nowrap;
}
.tree-connector {
    color: #C8BFA8;
    user-select: none;
    min-width: 14px;
}
.tree-rule {
    color: #0A0A0A;
    font-weight: 500;
}
.tree-rule strong { font-weight: 700; color: #0A0A0A; }
.tree-leaf-approved {
    background: #0A0A0A;
    color: #F5F0E8;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.1rem 0.5rem;
    font-weight: 700;
}
.tree-leaf-rejected {
    background: #F5F0E8;
    color: #0A0A0A;
    border: 1.5px solid #0A0A0A;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.1rem 0.5rem;
    font-weight: 700;
}

/* ── Stat box (dataset tab) ── */
.stat-box-row {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 1px;
    border: 1.5px solid #0A0A0A;
    background: #0A0A0A;
    margin-bottom: 2rem;
}
.stat-box {
    background: #FDFAF4;
    padding: 1.2rem 1.5rem;
}
.stat-box-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0A0A0A;
    display: block;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-box-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
}

/* ── Result verdict ── */
.verdict-approved {
    border: 2px solid #0A0A0A;
    padding: 2.5rem;
    margin-bottom: 2rem;
    background: #0A0A0A;
    color: #F5F0E8;
    position: relative;
}
.verdict-rejected {
    border: 2px solid #0A0A0A;
    padding: 2.5rem;
    margin-bottom: 2rem;
    background: #F5F0E8;
    color: #0A0A0A;
    position: relative;
}
.verdict-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    opacity: 0.6;
}
.verdict-text {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
}
.verdict-sub {
    margin-top: 0.8rem;
    font-size: 0.85rem;
    opacity: 0.7;
    line-height: 1.5;
}

/* ── Decision steps ── */
.steps-container {
    border: 1.5px solid #D0C8B8;
    background: #FDFAF4;
}
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.9rem 1.2rem;
    border-bottom: 1px solid #E8E2D6;
}
.step-row:last-child { border-bottom: none; }
.step-index {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #999;
    min-width: 36px;
    padding-top: 2px;
    letter-spacing: 0.1em;
}
.step-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #0A0A0A;
    line-height: 1.5;
}

/* ── Metric row ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1px;
    border: 1.5px solid #0A0A0A;
    background: #0A0A0A;
    margin-top: 1.5rem;
}
.metric-cell {
    background: #F5F0E8;
    padding: 1rem 1.2rem;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0A0A0A;
    display: block;
    margin-bottom: 0.15rem;
}
.metric-key {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #999;
}

/* ── Footer ── */
.site-footer {
    border-top: 2px solid #0A0A0A;
    padding: 1.2rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0A0A0A;
}
.footer-left {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #666;
    letter-spacing: 0.1em;
}
.footer-right {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #444;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Tabs (Explorer section) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid #0A0A0A !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 1.5rem !important;
    border-right: 1.5px solid #0A0A0A !important;
    color: #666 !important;
    background: #F5F0E8 !important;
}
.stTabs [aria-selected="true"] {
    background: #0A0A0A !important;
    color: #F5F0E8 !important;
    font-weight: 700 !important;
}

/* ── Code block ── */
.stCodeBlock {
    border: 1.5px solid #0A0A0A !important;
    border-radius: 0 !important;
}
code {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Hide Streamlit frame elements */
[data-testid="stDecoration"] { display: none; }
[data-testid="collapsedControl"] { display: none !important; }
.stSpinner > div { border-color: #0A0A0A transparent transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Model (inlined) ────────────────────────────────────────────────────────────
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
            op = "≤" if fval <= tval else ">"
            reasons.append(f"{fname}  {op}  {tval:.1f}  (yours: {disp})")
        return status, reasons

# ── Cache ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model_and_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "loan_approval_dataset.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    if "loan_id" in df.columns:
        df = df.drop("loan_id", axis=1)
    model = LoanModel()
    acc, rules = model.train(df)
    return model, df, acc, rules

with st.spinner(""):
    model, df, accuracy, tree_rules = get_model_and_data()

total    = len(df)
approved = (df["loan_status"] == "Approved").sum()
rejected = total - approved

# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nav-bar">
  <span class="nav-logo">◈ LOANIQ</span>
  <span class="nav-tag">Explainable AI · Decision Tree</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">AI-powered loan analysis</div>
  <div class="hero-title">
    <span>KNOW WHY</span>
    <span class="accent">YOUR LOAN</span>
    <span>WAS REJECTED.</span>
  </div>
  <p class="hero-desc">
    Transparent, rule-based decisions. No black boxes. 
    Every outcome traced to its exact logical path — readable by anyone.
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STAT STRIP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="stat-strip">
  <div class="stat-item">
    <span class="stat-num">{total:,}</span>
    <span class="stat-label">Training Records</span>
  </div>
  <div class="stat-item">
    <span class="stat-num">{accuracy*100:.1f}%</span>
    <span class="stat-label">Model Accuracy</span>
  </div>
  <div class="stat-item">
    <span class="stat-num">{approved/total*100:.0f}%</span>
    <span class="stat-label">Approval Rate</span>
  </div>
  <div class="stat-item">
    <span class="stat-num">4</span>
    <span class="stat-label">Decision Levels</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN GRID: FORM  |  RESULT
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 1], gap="small")

with left_col:
    st.markdown('<div class="section-label">Applicant Information</div>', unsafe_allow_html=True)

    with st.form("loan_form", clear_on_submit=False):

        st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1:
            income = st.number_input("Annual Income (₹)", min_value=0, max_value=100_000_000,
                                     value=500_000, step=10_000, label_visibility="visible")
        with f2:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, max_value=200_000_000,
                                          value=1_000_000, step=10_000)

        f3, f4 = st.columns(2)
        with f3:
            loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=12)
        with f4:
            cibil = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)

        st.markdown('<div class="section-label">Profile & Employment</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with p2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        with p3:
            dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0)

        st.markdown('<div class="section-label">Asset Portfolio</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            bank_asset = st.number_input("Bank Assets (₹)", min_value=0, value=0, step=10_000)
            com_asset  = st.number_input("Commercial Assets (₹)", min_value=0, value=0, step=10_000)
        with a2:
            res_asset  = st.number_input("Residential Assets (₹)", min_value=0, value=0, step=10_000)
            lux_asset  = st.number_input("Luxury Assets (₹)", min_value=0, value=0, step=10_000)

        submitted = st.form_submit_button("→  ANALYSE APPLICATION", use_container_width=True)

with right_col:
    st.markdown('<div class="section-label">Decision Output</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div style="border:1.5px dashed #C8BFA8;padding:3rem 2rem;text-align:center;color:#AAA;">
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:0.15em;
                        text-transform:uppercase;margin-bottom:0.8rem;">Awaiting Input</div>
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">◈</div>
            <div style="font-size:0.85rem;line-height:1.6;color:#BBB;">
                Fill the form and submit<br>to see your decision path.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
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
        status, reasons = model.predict(form_data)
        is_approved = status == "Approved"

        if is_approved:
            st.markdown(f"""
            <div class="verdict-approved">
              <div class="verdict-badge">◈ Decision Output</div>
              <div class="verdict-text">APPROVED</div>
              <div class="verdict-sub">
                Your application satisfies the lending criteria.<br>
                {len(reasons)} decision rule(s) evaluated.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-rejected">
              <div class="verdict-badge">◈ Decision Output</div>
              <div class="verdict-text">REJECTED</div>
              <div class="verdict-sub">
                Your application did not meet the required criteria.<br>
                {len(reasons)} decision rule(s) evaluated.
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Decision Trace — Rule Path</div>', unsafe_allow_html=True)
        steps_html = '<div class="steps-container">'
        for i, step in enumerate(reasons, 1):
            steps_html += f"""
            <div class="step-row">
              <span class="step-index">S{i:02d}</span>
              <span class="step-text">{step}</span>
            </div>"""
        steps_html += '</div>'
        st.markdown(steps_html, unsafe_allow_html=True)

        total_assets = bank_asset + res_asset + com_asset + lux_asset
        cibil_status = "GOOD" if cibil > 700 else ("FAIR" if cibil > 550 else "POOR")
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-cell">
            <span class="metric-val">{cibil}</span>
            <span class="metric-key">CIBIL — {cibil_status}</span>
          </div>
          <div class="metric-cell">
            <span class="metric-val">₹{loan_amount/100000:.1f}L</span>
            <span class="metric-key">Loan Amount</span>
          </div>
          <div class="metric-cell">
            <span class="metric-val">₹{income/100000:.1f}L</span>
            <span class="metric-key">Annual Income</span>
          </div>
          <div class="metric-cell">
            <span class="metric-val">₹{total_assets/100000:.1f}L</span>
            <span class="metric-key">Total Assets</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER (below the fold)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-top:2px solid #0A0A0A;border-bottom:2px solid #0A0A0A;
            padding:1rem 2.5rem;background:#0A0A0A;">
  <span style="font-family:'Space Mono',monospace;font-size:0.7rem;
               letter-spacing:0.2em;text-transform:uppercase;color:#888;">
    // Data Explorer & Model Internals
  </span>
</div>
""", unsafe_allow_html=True)

with st.container():
    tab1, tab2, tab3 = st.tabs(["FEATURE WEIGHTS", "DECISION RULES", "DATASET"])

    with tab1:
        st.markdown("")
        imps = model.model.feature_importances_
        feat_df = (
            pd.DataFrame({"Feature": model.feature_names, "Weight": (imps * 100).round(2)})
            .sort_values("Weight", ascending=False)
            .reset_index(drop=True)
        )
        feat_df.index = [f"F{i+1:02d}" for i in range(len(feat_df))]
        st.dataframe(feat_df, use_container_width=True)
        st.bar_chart(feat_df.set_index("Feature")["Weight"])

    with tab2:
        st.markdown('<div class="section-label" style="margin:1.2rem 0 1rem;">How the model makes decisions — read top to bottom</div>', unsafe_allow_html=True)
        # Parse tree rules into styled HTML
        def render_tree(rules_text):
            lines = rules_text.strip().split('\n')
            html = '<div class="tree-container">'
            for line in lines:
                raw = line
                # count leading pipe/space chars for indent
                stripped = raw.lstrip()
                leading  = raw[: len(raw) - len(stripped)]
                depth    = leading.count('|')
                indent_px = depth * 20
                connector = raw.replace(stripped, '').replace('|', '┊').replace('-', '') if depth else ''
                if 'class: 1' in stripped:
                    inner = '<span class="tree-leaf-approved">✓ APPROVED</span>'
                elif 'class: 0' in stripped:
                    inner = '<span class="tree-leaf-rejected">✕ REJECTED</span>'
                else:
                    # highlight the feature name
                    cond = stripped.lstrip('|- ')
                    parts = cond.split(' ', 1)
                    feat  = f'<strong>{parts[0]}</strong>' if parts else cond
                    rest  = ' ' + parts[1] if len(parts) > 1 else ''
                    inner = f'<span class="tree-rule">{feat}{rest}</span>'
                conn_html = f'<span class="tree-connector" style="padding-left:{indent_px}px">{'└─' if '|---' in raw else '┊ '}</span>'
                html += f'<div class="tree-line">{conn_html}{inner}</div>'
            html += '</div>'
            return html
        st.markdown(render_tree(tree_rules), unsafe_allow_html=True)

    with tab3:
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        # Custom stat boxes — no st.metric() blue tint
        st.markdown(f"""
        <div class="stat-box-row">
          <div class="stat-box">
            <span class="stat-box-num">{total:,}</span>
            <span class="stat-box-label">Total Records</span>
          </div>
          <div class="stat-box">
            <span class="stat-box-num">{approved:,}</span>
            <span class="stat-box-label">Approved</span>
          </div>
          <div class="stat-box">
            <span class="stat-box-num">{rejected:,}</span>
            <span class="stat-box-label">Rejected</span>
          </div>
          <div class="stat-box">
            <span class="stat-box-num">{approved/total*100:.1f}%</span>
            <span class="stat-box-label">Approval Rate</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        left_c, right_c = st.columns(2)
        with left_c:
            st.markdown('<div class="section-label">CIBIL Score Distribution by Status</div>', unsafe_allow_html=True)
            cibil_chart = (
                df.groupby(["cibil_score", "loan_status"])
                  .size().unstack(fill_value=0)
            )
            st.line_chart(cibil_chart, use_container_width=True)
        with right_c:
            st.markdown('<div class="section-label">Loan Amount by Outcome</div>', unsafe_allow_html=True)
            df2   = df.copy()
            bins  = pd.cut(df2["loan_amount"], bins=12)
            chart = df2.groupby([bins, "loan_status"]).size().unstack(fill_value=0)
            chart.index = chart.index.astype(str)
            st.bar_chart(chart, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem">Raw Dataset Sample</div>', unsafe_allow_html=True)
        n = st.slider("Number of rows", min_value=5, max_value=100, value=15, step=5)
        st.dataframe(
            df.head(n).style.set_properties(**{
                'background-color': '#FDFAF4',
                'color': '#0A0A0A',
                'border-color': '#D0C8B8',
                'font-family': 'Space Grotesk, sans-serif',
                'font-size': '13px',
            }),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="site-footer">
  <span class="footer-left">◈ LoanIQ — Explainable Loan Decisions · Built with Decision Trees</span>
  <span class="footer-right">sklearn · streamlit · open source</span>
</div>
""", unsafe_allow_html=True)
