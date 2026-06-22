# Explainable Loan Rejection Simulator

An AI-powered loan decision simulator with full transparency — see the exact decision rules that led to every outcome.

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/Raj1952/Modern-Loan-Prediction/main/streamlit_app.py)

## Features
- 📋 **Interactive Loan Form** — Sliders, dropdowns, and number inputs for all applicant details
- 🧠 **Decision Path Explanation** — Step-by-step IF-THEN rules behind every decision
- 🌳 **Global Model Rules** — Full decision tree text + feature importances
- 📊 **Dataset Explorer** — Charts showing CIBIL score & loan amount distributions

## Tech Stack
- **Frontend:** Streamlit
- **ML Model:** Decision Tree Classifier (scikit-learn, max_depth=4)
- **Dataset:** Indian Loan Approval Dataset

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
