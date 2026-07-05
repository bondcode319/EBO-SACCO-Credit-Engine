"""
================================================================================
EBO-SACCO Credit Scoring System — Model Evaluation (v2.0)
================================================================================
Evaluates all trained models on the test set with full metrics,
confusion matrices, McNemar's test, and SHAP explainability.

Usage:
    python eval_model.py
    python eval_model.py --data "path/to/dataset.xlsx"
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import pickle
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from scipy import stats

from config import (
    BASE_DIR, DATA_FILE, MODELS_DIR, PLOTS_DIR, LOGS_DIR,
    RANDOM_STATE, FIG_DPI, COLORS
)

# Logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'evaluation.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_models():
    """Load all trained model artifacts from MODELS_DIR."""
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
            logger.info(f"Loaded: {name}")
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    return models, metadata


def main():
    parser = argparse.ArgumentParser(description='EBO-SACCO Evaluation v2.0')
    parser.add_argument('--data', type=str, default=DATA_FILE)
    args = parser.parse_args()

    # Rebuild the exact test split using the training pipeline
    from train import load_data, preprocess, split_data
    df = load_data(args.data)
    df = preprocess(df)
    data = split_data(df)
    X_test, y_test = data['X_test'], data['y_test']
    feature_names = data['feature_names']

    models, metadata = load_models()
    best_name = metadata.get('best_model')
    logger.info(f"Best model per metadata: {best_name}")

    # ── Metrics + McNemar's test vs best model ────────────────────────────────
    preds = {}
    for name, model in models.items():
        yp = model.predict(X_test)
        ypr = model.predict_proba(X_test)[:, 1]
        preds[name] = yp
        logger.info(
            f"{name}: Acc={accuracy_score(y_test, yp):.4f} "
            f"Prec={precision_score(y_test, yp, zero_division=0):.4f} "
            f"Rec={recall_score(y_test, yp, zero_division=0):.4f} "
            f"F1={f1_score(y_test, yp, zero_division=0):.4f} "
            f"AUC={roc_auc_score(y_test, ypr):.4f}")

    if best_name in preds:
        bp = preds[best_name]
        for other, op in preds.items():
            if other == best_name:
                continue
            a = int(sum((bp == y_test) & (op != y_test)))
            b = int(sum((op == y_test) & (bp != y_test)))
            if a + b > 0:
                chi2 = (abs(a - b) - 1) ** 2 / (a + b)
                pv = 1 - stats.chi2.cdf(chi2, df=1)
                sig = "SIGNIFICANT" if pv < 0.05 else "not significant"
                logger.info(f"McNemar {best_name} vs {other}: "
                            f"chi2={chi2:.4f}, p={pv:.6f} ({sig})")

    # ── SHAP explainability for the best model ────────────────────────────────
    try:
        import shap
        model = models[best_name]
        sample = X_test[:500]  # cap for speed
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        os.makedirs(PLOTS_DIR, exist_ok=True)
        shap.summary_plot(shap_values, sample, feature_names=feature_names,
                          show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '06_shap_summary.png'), dpi=FIG_DPI)
        plt.close()

        shap.summary_plot(shap_values, sample, feature_names=feature_names,
                          plot_type='bar', show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '07_shap_bar.png'), dpi=FIG_DPI)
        plt.close()

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
        shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
        shap_df.to_csv(os.path.join(PLOTS_DIR, 'shap_feature_importance.csv'), index=False)
        logger.info("SHAP plots and feature importance saved")
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")

    logger.info("Evaluation complete")


if __name__ == '__main__':
    main()