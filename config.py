"""
================================================================================
EBO-SACCO Credit Scoring System — Centralized Configuration
================================================================================
All paths, feature lists, hyperparameters, and constants in one place.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Loan History_Dataset (1).xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ── Random State ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Temporal Split ────────────────────────────────────────────────────────────
TEST_YEAR = 2025  # train: 2021-2024, test: 2025
FALLBACK_TEST_SIZE = 0.2  # used if temporal split yields < 500 test samples

# ── Cross Validation ──────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Feature Configuration ────────────────────────────────────────────────────
# Columns to drop (identifiers, zero-variance, post-disbursement leakage)
DROP_COLUMNS = [
    'BorrowerID', 'Name',
    # Zero variance
    'MobileMoneyTransactions',
    # Note: Post-disbursement features (MissedInstallments, DaysInArrears,
    # NumberOfRepayments, AverageRepaymentValue, NumberOfDeposits, AccountActivity)
    # are INCLUDED for research purposes (portfolio monitoring use case).
    # For pre-approval-only scoring, uncomment lines below:
    # 'MissedInstallments', 'DaysInArrears', 'NumberOfRepayments',
    # 'AverageRepaymentValue', 'NumberOfDeposits', 'AccountActivity',
]

# Date columns used for temporal features then dropped
DATE_COLUMNS = ['ApprovalDate', 'FirstDisbursementDate']

# High-cardinality categorical — group into top N + 'Other'
LOAN_TYPE_TOP_N = 10

# Marital status grouping
MARITAL_MAP = {
    'MARRIED': 'MARRIED', 'SINGLE': 'SINGLE',
    'WIDOW OR WIDOWER': 'OTHER', 'COHABITING': 'OTHER',
    'SEPARATED': 'OTHER', 'DIVORCED': 'OTHER', 'COUPLE': 'MARRIED'
}

# Outlier capping columns (99th percentile)
OUTLIER_CAP_COLUMNS = [
    'AverageMonthlyIncome', 'CurrentSavingsBalance',
    'LoanAmount', 'TotalCollateralValue', 'AverageRepaymentValue'
]

# ── Model Hyperparameters ─────────────────────────────────────────────────────
HYPERPARAMS = {
    'Logistic Regression': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
    },
    'Decision Tree': {
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'class_weight': 'balanced',
    },
    'Random Forest': {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'n_jobs': -1,
    },
    'XGBoost': {
        'n_estimators': 250,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'n_jobs': -1,
    },
}

# ── API Configuration ─────────────────────────────────────────────────────────
API_PORT = 5000
ALLOWED_ORIGINS = ["http://localhost:5000", "http://127.0.0.1:5000",
                   "http://localhost:8501", "http://127.0.0.1:8501"]

# Business rule guardrails
MIN_AGE = 18
MAX_AGE = 75
MAX_DTI = 60.0
MAX_LTV = 100.0
SAVINGS_MULTIPLIER = 3  # max loan = 3x savings

# Input validation bounds
INPUT_BOUNDS = {
    'age': (18, 120),
    'income': (0, 1e9),
    'savings': (0, 1e9),
    'loan_amount': (1, 1e9),
    'collateral': (0, 1e9),
    'loan_duration': (1, 72),
    'guarantor_count': (0, 20),
    'membership_years': (0, 50),
    'previous_loans_count': (0, 100),
    'interest_rate': (0, 100),
}

# ── Visualization ─────────────────────────────────────────────────────────────
FIG_DPI = 150
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

# ── Model Version ─────────────────────────────────────────────────────────────
MODEL_VERSION = "2.0.0"
