"""
================================================================================
EBO-SACCO Credit Scoring System — Flask API (v2.0)
================================================================================
REST API for loan approval predictions with SHAP explainability.

Improvements over v1.0:
  - SHAP import at module level (not per-request)
  - Per-model-type SHAP explainer (TreeExplainer / LinearExplainer)
  - Specific exception handling (no bare except)
  - Restricted CORS origins
  - Input validation with bounds
  - Correct confidence score (P(default), not max(proba))
  - 3x savings guardrail implemented
  - Logging with file handler
  - Relative paths via config.py

Usage:
    python api.py
================================================================================
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import (
    BASE_DIR, MODELS_DIR, API_PORT, ALLOWED_ORIGINS, LOGS_DIR,
    MIN_AGE, MAX_AGE, MAX_DTI, MAX_LTV, SAVINGS_MULTIPLIER, INPUT_BOUNDS
)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'api.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS)

# ── Load Models at Startup ────────────────────────────────────────────────────
def load_all():
    """Load models, scaler, feature names, and metadata at startup."""
    model_files = {
        'Logistic Regression': 'credit_model_Logistic_Regression.pkl',
        'Decision Tree': 'credit_model_Decision_Tree.pkl',
        'Random Forest': 'credit_model_Random_Forest.pkl',
        'XGBoost': 'credit_model_XGBoost.pkl',
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model not found: {path}")

    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    features_path = os.path.join(MODELS_DIR, 'feature_names.json')
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    return models, scaler, feature_names, metadata


models, scaler, feature_names, model_metadata = load_all()

# ── Pre-build SHAP explainers (one per model type) ───────────────────────────
shap_explainers = {}
for name, model in models.items():
    try:
        model_type = type(model).__name__
        if model_type in ['RandomForestClassifier', 'XGBClassifier',
                          'DecisionTreeClassifier']:
            shap_explainers[name] = shap.TreeExplainer(model)
        elif model_type == 'LogisticRegression':
            shap_explainers[name] = None  # deferred — needs background data
        logger.info(f"SHAP explainer ready: {name} ({model_type})")
    except Exception as e:
        logger.warning(f"SHAP explainer failed for {name}: {e}")
        shap_explainers[name] = None


# ── Input Validation ──────────────────────────────────────────────────────────
REQUIRED_FIELDS = [
    'age', 'employment_status', 'prev_default', 'income', 'savings',
    'loan_amount', 'collateral', 'loan_duration', 'guarantor_count',
    'membership_years', 'previous_loans_count', 'interest_rate'
]

NUMERIC_FIELDS = {
    'age': 'age', 'income': 'income', 'savings': 'savings',
    'loan_amount': 'loan_amount', 'collateral': 'collateral',
    'loan_duration': 'loan_duration', 'guarantor_count': 'guarantor_count',
    'membership_years': 'membership_years',
    'previous_loans_count': 'previous_loans_count',
    'interest_rate': 'interest_rate',
}


def validate_input(data):
    """Validate request payload. Returns (parsed_dict, error_message)."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object"

    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in (None, '')]
    if missing:
        return None, f"Missing required fields: {', '.join(missing)}"

    parsed = {}
    for field, bounds_key in NUMERIC_FIELDS.items():
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            return None, f"Field '{field}' must be a number"
        lo, hi = INPUT_BOUNDS[bounds_key]
        if not (lo <= value <= hi):
            return None, f"Field '{field}' out of bounds [{lo}, {hi}]"
        parsed[field] = value

    parsed['employment_status'] = str(data['employment_status'])
    parsed['prev_default'] = str(data['prev_default'])
    parsed['model_choice'] = str(data.get('model_choice', model_metadata.get('best_model', 'XGBoost')))
    return parsed, None


# ── Feature Vector Construction ───────────────────────────────────────────────
# Post-disbursement features are unknown at application time. Zero or mean
# values land in low-density regions of these skewed distributions and push
# every applicant toward default, so impute the training-set MEDIAN (a dense,
# class-neutral point). Values computed from the v2.0.0 training data —
# regenerate if the model is retrained on new data.
UNKNOWN_AT_APPLICATION = {
    'MissedInstallments': 11.0,
    'DaysInArrears': 0.0,
    'NumberOfRepayments': 1.0,
    'AverageRepaymentValue': 128750.0,
    'NumberOfDeposits': 65.0,
    'AccountActivity': 38.0,
}


def build_feature_vector(p):
    """Build a single-row DataFrame matching the 43 training features.

    Post-disbursement features are zeroed — a new applicant has no repayment
    history yet. One-hot groups default to the training base categories
    (RepaymentSchedule=MONTHLY, MaritalStatus=MARRIED, LoanType base).
    """
    now = datetime.now()
    income = p['income']
    loan = p['loan_amount']
    collateral = p['collateral']
    savings = p['savings']
    duration = p['loan_duration']

    dti = (loan / duration) / income * 100 if income > 0 else 0.0
    ltv = loan / collateral * 100 if collateral > 0 else 0.0

    row = {name: 0.0 for name in feature_names}
    row.update({
        'Gender': 0,  # not collected (ethical compliance)
        'Age': p['age'],
        'EmploymentStatus': 1 if p['employment_status'] == 'Self-Employed' else 0,
        'TotalCollateralValue': collateral,
        'GuarantorCount': p['guarantor_count'],
        'AverageMonthlyIncome': income,
        'CurrentSavingsBalance': savings,
        'MembershipYears': p['membership_years'],
        'PreviousLoansCount': p['previous_loans_count'],
        'PreviousDefaultHistory': 1 if p['prev_default'] == 'Yes' else 0,
        'LoanAmount': loan,
        'InterestRate': p['interest_rate'],
        'LoanDuration': duration,
        'ApprovalYear': now.year,
        'ApprovalMonth': now.month,
        'ApprovalQuarter': (now.month - 1) // 3 + 1,
        'DisbursementDelay': 0,
        'LoanToIncomeRatio': loan / income if income > 0 else 0.0,
        'LoanToCollateralRatio': loan / collateral if collateral > 0 else 0.0,
        'SavingsToLoanRatio': savings / loan if loan > 0 else 0.0,
        'CollateralCoverage': collateral / loan if loan > 0 else 0.0,
        'DTI': dti,
        'LTV': ltv,
        'RepaymentSchedule_MONTHLY': 1,
    })
    X = pd.DataFrame([row], columns=feature_names)
    for col, median_value in UNKNOWN_AT_APPLICATION.items():
        if col in feature_names:
            X[col] = median_value
    return X, dti, ltv


# ── Business Rule Guardrails ──────────────────────────────────────────────────
def check_guardrails(p, dti, ltv):
    """Return list of violated business rules (empty = pass)."""
    violations = []
    if not (MIN_AGE <= p['age'] <= MAX_AGE):
        violations.append(f"Applicant age must be between {MIN_AGE} and {MAX_AGE}")
    if dti > MAX_DTI:
        violations.append(f"Debt-to-income ratio {dti:.1f}% exceeds maximum {MAX_DTI}%")
    if ltv > MAX_LTV:
        violations.append(f"Loan-to-value ratio {ltv:.1f}% exceeds maximum {MAX_LTV}%")
    if p['loan_amount'] > SAVINGS_MULTIPLIER * p['savings']:
        violations.append(
            f"Loan exceeds {SAVINGS_MULTIPLIER}x savings balance "
            f"(max UGX {SAVINGS_MULTIPLIER * p['savings']:,.0f})")
    return violations


# ── SHAP Explanations ─────────────────────────────────────────────────────────
def explain_prediction(model_name, X_scaled_df, top_n=5):
    """Top-N decision drivers as [{feature, impact, direction}]."""
    try:
        explainer = shap_explainers.get(model_name)
        if explainer is not None:
            shap_values = explainer.shap_values(X_scaled_df.values)
            # RF/DT may return per-class list or 3D array; take class 1 (Repaid)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            contributions = shap_values[0]
        else:
            model = models[model_name]
            if not hasattr(model, 'coef_'):
                return []
            contributions = model.coef_[0] * X_scaled_df.values[0]

        # Rank by |contribution|, skipping features unknown at application time
        order = np.argsort(np.abs(contributions))[::-1]
        results = []
        for i in order:
            if feature_names[i] in UNKNOWN_AT_APPLICATION:
                continue
            results.append({
                'feature': feature_names[i],
                'impact': round(float(abs(contributions[i])), 4),
                # positive contribution pushes toward class 1 (Repaid)
                'direction': 'decreases risk' if contributions[i] > 0 else 'increases risk',
            })
            if len(results) >= top_n:
                break
        return results
    except (ValueError, IndexError, AttributeError) as e:
        logger.warning(f"SHAP explanation failed for {model_name}: {e}")
        return []


# ── Routes ────────────────────────────────────────────────────────────────────
WEBAPP_DIR = os.path.join(BASE_DIR, 'webapp')


@app.route('/')
def serve_index():
    from flask import send_from_directory
    return send_from_directory(WEBAPP_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    from flask import send_from_directory
    return send_from_directory(WEBAPP_DIR, filename)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'best_model': model_metadata.get('best_model'),
        'version': model_metadata.get('version'),
    })


@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({
        'models': list(models.keys()),
        'best_model': model_metadata.get('best_model'),
        'test_metrics': model_metadata.get('test_metrics', {}),
    })


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    parsed, error = validate_input(data)
    if error:
        logger.warning(f"Validation error: {error}")
        return jsonify({'message': error}), 400

    model_name = parsed['model_choice']
    if model_name not in models:
        return jsonify({'message': f"Unknown model '{model_name}'. "
                        f"Available: {list(models.keys())}"}), 400

    X_raw, dti, ltv = build_feature_vector(parsed)
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=feature_names)

    model = models[model_name]
    proba = model.predict_proba(X_scaled.values)[0]
    p_default = float(proba[0])   # class 0 = Default
    p_repaid = float(proba[1])    # class 1 = Repaid
    model_approves = p_repaid >= 0.5

    violations = check_guardrails(parsed, dti, ltv)

    if violations:
        is_approved = False
        status = 'DENIED — POLICY RULE'
        confidence = round(p_default * 100, 1)
    elif model_approves:
        is_approved = True
        status = 'LOAN APPROVED'
        confidence = round(p_repaid * 100, 1)
    else:
        is_approved = False
        status = 'LOAN DENIED — HIGH RISK'
        confidence = round(p_default * 100, 1)

    explanations = explain_prediction(model_name, X_scaled)

    logger.info(f"Prediction: model={model_name} approved={is_approved} "
                f"P(default)={p_default:.3f} DTI={dti:.1f} LTV={ltv:.1f} "
                f"violations={len(violations)}")

    return jsonify({
        'is_approved': is_approved,
        'status': status,
        'confidence': confidence,
        'default_probability': round(p_default * 100, 1),
        'repaid_probability': round(p_repaid * 100, 1),
        'model_used': model_name,
        'dti': round(dti, 1),
        'ltv': round(ltv, 1),
        'guardrail_violations': violations,
        'explanations': explanations,
        'timestamp': datetime.now().isoformat(),
    })


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info(f"EBO-SACCO API v{model_metadata.get('version', '?')} starting on port {API_PORT}")
    logger.info(f"Webapp served at http://localhost:{API_PORT}/")
    app.run(host='127.0.0.1', port=API_PORT, debug=False)