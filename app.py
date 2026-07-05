"""
================================================================================
EBO-SACCO Credit Scoring System — Streamlit Frontend (v2.0)
================================================================================
Consistent with api.py: same features, same models, same preprocessing.

Improvements over v1.0:
  - Gender removed (ethical compliance)
  - Loads all 4 models with selector
  - Aligned feature set with api.py
  - Uses models/ directory (relative paths)
  - Correct confidence score
  - 3x savings guardrail implemented

Usage:
    streamlit run app.py
================================================================================
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    MODELS_DIR, MIN_AGE, MAX_AGE, MAX_DTI, MAX_LTV, SAVINGS_MULTIPLIER
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EBO-SACCO Credit System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-color: #f4f4f0;
        --text-color: #111111;
        --accent-color: #0000ff;
        --border-color: #111111;
        --success-bg: #dcfce7;
        --success-border: #166534;
        --danger-bg: #fee2e2;
        --danger-border: #991b1b;
    }
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: "Courier New", Courier, monospace;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: var(--text-color) !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: -0.5px;
    }
    .main-header {
        border-bottom: 4px solid var(--border-color);
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    .main-title { font-size: 2.5rem !important; margin: 0; }
    .sub-title {
        font-family: "Courier New", Courier, monospace !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        color: #555555 !important;
        margin-top: 5px;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 2px solid var(--border-color);
        border-radius: 0px;
        padding: 20px;
        box-shadow: 4px 4px 0px 0px var(--border-color);
    }
    .stButton>button {
        background-color: var(--text-color);
        color: #ffffff;
        border: none;
        border-radius: 0px;
        padding: 0.75rem 1.5rem;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
    }
    .stButton>button:hover { background-color: var(--accent-color); }
    .result-card {
        padding: 20px;
        border: 2px solid var(--border-color);
        background-color: #ffffff;
        text-align: left;
        box-shadow: 4px 4px 0px 0px var(--border-color);
        margin-bottom: 20px;
    }
    .result-approved {
        background-color: var(--success-bg);
        border-color: var(--success-border);
        box-shadow: 4px 4px 0px 0px var(--success-border);
        color: var(--success-border);
    }
    .result-denied {
        background-color: var(--danger-bg);
        border-color: var(--danger-border);
        box-shadow: 4px 4px 0px 0px var(--danger-border);
        color: var(--danger-border);
    }
    .result-heading {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 1.5rem; font-weight: 800; margin-bottom: 10px;
        text-transform: uppercase;
    }
    .result-sub {
        font-family: "Courier New", Courier, monospace;
        font-weight: bold; font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 class="main-title">EBO-SACCO CREDIT SCORING SYSTEM</h1>
    <div class="sub-title">VERSION 2.0 // MACHINE LEARNING POWERED</div>
</div>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_files = {
        'Logistic Regression': 'credit_model_Logistic_Regression.pkl',
        'Decision Tree': 'credit_model_Decision_Tree.pkl',
        'Random Forest': 'credit_model_Random_Forest.pkl',
        'XGBoost': 'credit_model_XGBoost.pkl',
    }
    loaded = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                loaded[name] = pickle.load(f)

    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler_obj = pickle.load(f)

    features_path = os.path.join(MODELS_DIR, 'feature_names.json')
    with open(features_path, 'r') as f:
        feat_names = json.load(f)

    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    return loaded, scaler_obj, feat_names, meta


all_models, scaler, feature_names, metadata = load_models()

if not all_models:
    st.error("No models found. Run train.py first.")
    st.stop()

best_model_name = metadata.get('best_model', list(all_models.keys())[0])

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### APPLICANT DATA ENTRY")
    with st.form("prediction_form"):
        fc1, fc2 = st.columns(2)

        with fc1:
            age = st.number_input("AGE", min_value=18, max_value=120, value=30)
            employment_status = st.selectbox(
                "EMPLOYMENT STATUS", ["Employed", "Self-Employed"])
            prev_default = st.selectbox("PREVIOUS DEFAULT", ["No", "Yes"])
            income = st.number_input(
                "MONTHLY INCOME (UGX)", min_value=0.0, value=1000000.0,
                step=50000.0)
            savings = st.number_input(
                "SAVINGS BALANCE (UGX)", min_value=0.0, value=500000.0,
                step=50000.0)
            guarantor_count = st.number_input(
                "GUARANTOR COUNT", min_value=0, max_value=20, value=1)

        with fc2:
            loan_amount = st.number_input(
                "LOAN AMOUNT (UGX)", min_value=10000.0, value=2000000.0,
                step=100000.0)
            loan_duration = st.number_input(
                "LOAN DURATION (MONTHS)", min_value=1, max_value=72, value=12)
            collateral = st.number_input(
                "COLLATERAL VALUE (UGX)", min_value=0.0, value=3000000.0,
                step=100000.0)
            membership_years = st.number_input(
                "MEMBERSHIP YEARS", min_value=0, max_value=50, value=3)
            previous_loans = st.number_input(
                "PREVIOUS LOANS COUNT", min_value=0, max_value=100, value=1)
            interest_rate = st.number_input(
                "INTEREST RATE (%)", min_value=0.0, max_value=100.0,
                value=24.0)

        model_choice = st.selectbox(
            "AI ALGORITHM",
            list(all_models.keys()),
            index=list(all_models.keys()).index(best_model_name)
            if best_model_name in all_models else 0)

        submit_button = st.form_submit_button("PROCESS APPLICATION")

with col2:
    st.markdown("### SYSTEM OUTPUT")

    if submit_button:
        # Build feature vector
        emp_binary = 1 if employment_status == 'Self-Employed' else 0
        default_binary = 1 if prev_default == 'Yes' else 0

        from datetime import datetime
        now = datetime.now()

        dti = (loan_amount / loan_duration) / income * 100 if income > 0 else 0.0
        ltv = loan_amount / collateral * 100 if collateral > 0 else 0.0

        row = {name: 0.0 for name in feature_names}
        row.update({
            'Gender': 0,  # not collected (ethical compliance)
            'Age': age,
            'EmploymentStatus': emp_binary,
            'TotalCollateralValue': collateral,
            'GuarantorCount': guarantor_count,
            'AverageMonthlyIncome': income,
            'CurrentSavingsBalance': savings,
            'MembershipYears': membership_years,
            'PreviousLoansCount': previous_loans,
            'PreviousDefaultHistory': default_binary,
            'LoanAmount': loan_amount,
            'InterestRate': interest_rate,
            'LoanDuration': loan_duration,
            'ApprovalYear': now.year,
            'ApprovalMonth': now.month,
            'ApprovalQuarter': (now.month - 1) // 3 + 1,
            'DisbursementDelay': 0,
            'LoanToIncomeRatio': loan_amount / income if income > 0 else 0.0,
            'LoanToCollateralRatio': loan_amount / collateral if collateral > 0 else 0.0,
            'SavingsToLoanRatio': savings / loan_amount if loan_amount > 0 else 0.0,
            'CollateralCoverage': collateral / loan_amount if loan_amount > 0 else 0.0,
            'DTI': dti,
            'LTV': ltv,
            'RepaymentSchedule_MONTHLY': 1,
        })
        X_raw = pd.DataFrame([row], columns=feature_names)
        # Post-disbursement features are unknown at application time — impute
        # training-set medians (class-neutral); zero/mean values push every
        # applicant toward default. Same values as api.py.
        for col, med in {'MissedInstallments': 11.0, 'DaysInArrears': 0.0,
                         'NumberOfRepayments': 1.0, 'AverageRepaymentValue': 128750.0,
                         'NumberOfDeposits': 65.0, 'AccountActivity': 38.0}.items():
            if col in feature_names:
                X_raw[col] = med
        X_scaled = scaler.transform(X_raw)

        # Business rule guardrails
        violations = []
        if not (MIN_AGE <= age <= MAX_AGE):
            violations.append(f"Age must be between {MIN_AGE} and {MAX_AGE}")
        if dti > MAX_DTI:
            violations.append(f"DTI {dti:.1f}% exceeds maximum {MAX_DTI}%")
        if ltv > MAX_LTV:
            violations.append(f"LTV {ltv:.1f}% exceeds maximum {MAX_LTV}%")
        if loan_amount > SAVINGS_MULTIPLIER * savings:
            violations.append(
                f"Loan exceeds {SAVINGS_MULTIPLIER}x savings "
                f"(max UGX {SAVINGS_MULTIPLIER * savings:,.0f})")

        model = all_models[model_choice]
        proba = model.predict_proba(X_scaled)[0]
        p_default = float(proba[0])
        p_repaid = float(proba[1])
        model_approves = p_repaid >= 0.5

        if violations:
            st.markdown(f"""
            <div class="result-card result-denied">
                <div class="result-heading">DENIED — POLICY RULE</div>
                <div class="result-sub">Default probability: {p_default*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            for v in violations:
                st.error(v)
        elif model_approves:
            st.markdown(f"""
            <div class="result-card result-approved">
                <div class="result-heading">LOAN APPROVED</div>
                <div class="result-sub">Confidence: {p_repaid*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-denied">
                <div class="result-heading">LOAN DENIED — HIGH RISK</div>
                <div class="result-sub">Default probability: {p_default*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Risk metrics
        m1, m2 = st.columns(2)
        m1.metric("DTI RATIO", f"{dti:.1f}%",
                  delta="OK" if dti <= MAX_DTI else "HIGH",
                  delta_color="normal" if dti <= MAX_DTI else "inverse")
        m2.metric("LTV RATIO", f"{ltv:.1f}%",
                  delta="OK" if ltv <= MAX_LTV else "HIGH",
                  delta_color="normal" if ltv <= MAX_LTV else "inverse")
        m3, m4 = st.columns(2)
        m3.metric("P(DEFAULT)", f"{p_default*100:.1f}%")
        m4.metric("MODEL", model_choice)
    else:
        st.markdown("""
        <div class="result-card">
            <div class="result-heading">STANDBY</div>
            <div class="result-sub">Awaiting applicant data submission...</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer: Model Performance ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### MODEL PERFORMANCE (TEST SET)")
test_metrics = metadata.get('test_metrics', {})
if test_metrics:
    perf_df = pd.DataFrame(test_metrics).T.round(4)
    st.dataframe(perf_df, use_container_width=True)
st.caption(f"Model version {metadata.get('version', '?')} — trained {metadata.get('trained_at', '?')[:10]} — "
           f"best model: {metadata.get('best_model', '?')}")