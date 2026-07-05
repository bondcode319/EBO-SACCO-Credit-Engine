"""
EBO-SACCO Credit Scoring System - Training Pipeline v2.0
"""
import warnings
warnings.filterwarnings('ignore')
import os, sys, json, logging, pickle, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from scipy import stats
from config import (BASE_DIR, DATA_FILE, MODELS_DIR, PLOTS_DIR, LOGS_DIR,
    RANDOM_STATE, TEST_YEAR, FALLBACK_TEST_SIZE, CV_FOLDS,
    DROP_COLUMNS, DATE_COLUMNS, LOAN_TYPE_TOP_N, MARITAL_MAP,
    OUTLIER_CAP_COLUMNS, HYPERPARAMS, FIG_DPI, COLORS, MODEL_VERSION)

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, 'training.log')),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def load_data(path):
    logger.info("PHASE 1: DATA LOADING")
    df = pd.read_excel(path)
    df = df.dropna(subset=['LoanStatus'])
    df['LoanStatus'] = df['LoanStatus'].astype(int)
    logger.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"Default(0)={sum(df['LoanStatus']==0):,}, Repaid(1)={sum(df['LoanStatus']==1):,}")
    return df

def preprocess(df):
    logger.info("PHASE 2: PREPROCESSING & FEATURE ENGINEERING")
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns")
    if 'ApprovalDate' in df.columns:
        df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'], errors='coerce')
        df['ApprovalYear'] = df['ApprovalDate'].dt.year
        df['ApprovalMonth'] = df['ApprovalDate'].dt.month
        df['ApprovalQuarter'] = df['ApprovalDate'].dt.quarter
    if 'FirstDisbursementDate' in df.columns and 'ApprovalDate' in df.columns:
        df['FirstDisbursementDate'] = pd.to_datetime(df['FirstDisbursementDate'], errors='coerce')
        df['DisbursementDelay'] = (df['FirstDisbursementDate'] - df['ApprovalDate']).dt.days
        df['DisbursementDelay'] = df['DisbursementDelay'].fillna(0).clip(lower=0)
    df = df.drop(columns=[c for c in DATE_COLUMNS if c in df.columns])
    df['LoanToIncomeRatio'] = np.where(df['AverageMonthlyIncome'] > 0, df['LoanAmount'] / df['AverageMonthlyIncome'], 0)
    df['LoanToCollateralRatio'] = np.where(df['TotalCollateralValue'] > 0, df['LoanAmount'] / df['TotalCollateralValue'], 0)
    df['SavingsToLoanRatio'] = np.where(df['LoanAmount'] > 0, df['CurrentSavingsBalance'] / df['LoanAmount'], 0)
    df['CollateralCoverage'] = np.where(df['LoanAmount'] > 0, df['TotalCollateralValue'] / df['LoanAmount'], 0)
    df['DTI'] = np.where(df['AverageMonthlyIncome'] > 0, (df['LoanAmount'] / df['LoanDuration']) / df['AverageMonthlyIncome'] * 100, 0)
    df['LTV'] = np.where(df['TotalCollateralValue'] > 0, df['LoanAmount'] / df['TotalCollateralValue'] * 100, 0)
    logger.info("Engineered 6 ratio features")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    if 'LoanType' in df.columns:
        top_n = df['LoanType'].value_counts().nlargest(LOAN_TYPE_TOP_N).index
        df['LoanTypeGrouped'] = df['LoanType'].where(df['LoanType'].isin(top_n), 'Other')
        df = df.drop(columns=['LoanType'])
    if 'MaritalStatus' in df.columns:
        df['MaritalStatusGrouped'] = df['MaritalStatus'].map(MARITAL_MAP).fillna('OTHER')
        df = df.drop(columns=['MaritalStatus'])
    if 'Gender' in df.columns:
        df['Gender'] = (df['Gender'] == 'MALE').astype(int)
    if 'PreviousDefaultHistory' in df.columns:
        df['PreviousDefaultHistory'] = (df['PreviousDefaultHistory'] == 'Yes').astype(int)
    if 'EmploymentStatus' in df.columns:
        df['EmploymentStatus'] = (df['EmploymentStatus'] == 'SELF EMPLOYED').astype(int)
    cat_cols = [c for c in ['RepaymentSchedule', 'LoanTypeGrouped', 'MaritalStatusGrouped'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in OUTLIER_CAP_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(upper=df[col].quantile(0.99))
    for col in df.columns:
        if col != 'LoanStatus':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    logger.info(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns ({df.shape[1]-1} features)")
    return df

def split_data(df):
    logger.info("PHASE 3: TRAIN-TEST SPLIT")
    if 'ApprovalYear' in df.columns:
        train_mask = df['ApprovalYear'] <= (TEST_YEAR - 1)
        test_mask = df['ApprovalYear'] == TEST_YEAR
        n_test = test_mask.sum()
    else:
        n_test = 0
    if n_test >= 500:
        X_train = df[train_mask].drop(columns=['LoanStatus']).copy()
        y_train = df[train_mask]['LoanStatus'].values
        X_test = df[test_mask].drop(columns=['LoanStatus']).copy()
        y_test = df[test_mask]['LoanStatus'].values
        split_method = f"TEMPORAL (train: 2021-{TEST_YEAR-1}, test: {TEST_YEAR})"
    else:
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=['LoanStatus'])
        y = df['LoanStatus'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=FALLBACK_TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        split_method = "STRATIFIED 80/20"
    feature_names = list(X_train.columns)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)
    logger.info(f"Split: {split_method}")
    logger.info(f"Train: {X_train_scaled.shape[0]:,} | Test: {X_test_scaled.shape[0]:,} | SMOTE: {X_train_smote.shape[0]:,}")
    logger.info(f"Features: {len(feature_names)}")
    return {'X_train': X_train_scaled, 'X_train_smote': X_train_smote, 'X_test': X_test_scaled,
            'y_train': y_train, 'y_train_smote': y_train_smote, 'y_test': y_test,
            'feature_names': feature_names, 'scaler': scaler, 'split_method': split_method}

def train_models(data):
    logger.info("PHASE 4: MODEL TRAINING WITH CROSS-VALIDATION")
    X_train = data['X_train']
    X_train_smote = data['X_train_smote']
    y_train = data['y_train']
    y_train_smote = data['y_train_smote']
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = {}
    cv_scores = {}
    lr = LogisticRegression(**HYPERPARAMS['Logistic Regression'], random_state=RANDOM_STATE)
    scores = cross_val_score(lr, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores['Logistic Regression'] = scores
    lr.fit(X_train_smote, y_train_smote)
    models['Logistic Regression'] = lr
    logger.info(f"  LR  CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    dt = DecisionTreeClassifier(**HYPERPARAMS['Decision Tree'], random_state=RANDOM_STATE)
    scores = cross_val_score(dt, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores['Decision Tree'] = scores
    dt.fit(X_train_smote, y_train_smote)
    models['Decision Tree'] = dt
    logger.info(f"  DT  CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    rf = RandomForestClassifier(**HYPERPARAMS['Random Forest'], random_state=RANDOM_STATE)
    scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores['Random Forest'] = scores
    rf.fit(X_train_smote, y_train_smote)
    models['Random Forest'] = rf
    logger.info(f"  RF  CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    xgb_params = HYPERPARAMS['XGBoost'].copy()
    spw = sum(y_train == 0) / max(sum(y_train == 1), 1)
    xgb_params['scale_pos_weight'] = spw
    xgb = XGBClassifier(**xgb_params, use_label_encoder=False, random_state=RANDOM_STATE)
    scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores['XGBoost'] = scores
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    logger.info(f"  XGB CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return models, cv_scores

def evaluate_models(models, data, output_dir):
    logger.info("PHASE 5: EVALUATION ON TEST SET")
    X_test = data['X_test']
    y_test = data['y_test']
    results = {}
    for name, model in models.items():
        yp = model.predict(X_test)
        ypr = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, yp),
            'Precision': precision_score(y_test, yp, zero_division=0),
            'Recall (Repaid)': recall_score(y_test, yp, zero_division=0),
            'Recall (Default)': recall_score(y_test, yp, pos_label=0, zero_division=0),
            'F1-Score': f1_score(y_test, yp, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, ypr),
            'y_pred': yp, 'y_proba': ypr
        }
        logger.info(f"  {name}: Acc={results[name]['Accuracy']:.4f} F1={results[name]['F1-Score']:.4f} AUC={results[name]['ROC-AUC']:.4f}")
    metrics_cols = [k for k in list(results.values())[0].keys() if k not in ['y_pred', 'y_proba']]
    comp = pd.DataFrame({n: {k: v for k, v in r.items() if k in metrics_cols} for n, r in results.items()}).T.sort_values('ROC-AUC', ascending=False)
    best_name = comp['ROC-AUC'].idxmax()
    logger.info(f"BEST MODEL: {best_name} (AUC={comp.loc[best_name, 'ROC-AUC']:.4f})")
    best_pred = results[best_name]['y_pred']
    for other_name, other_r in results.items():
        if other_name == best_name:
            continue
        op = other_r['y_pred']
        a = sum((best_pred == y_test) & (op != y_test))
        b = sum((op == y_test) & (best_pred != y_test))
        if a + b > 0:
            chi2 = (abs(a - b) - 1) ** 2 / (a + b)
            pv = 1 - stats.chi2.cdf(chi2, df=1)
            logger.info(f"  McNemar {best_name} vs {other_name}: chi2={chi2:.4f}, p={pv:.6f}")
    comp.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    return results, comp, best_name

def generate_plots(results, comp, best_name, models, y_test, raw_data_path, output_dir):
    logger.info("PHASE 6: GENERATING VISUALIZATIONS")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    for (name, r), color in zip(results.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['ROC-AUC']:.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, '01_roc_curves.png'), dpi=FIG_DPI); plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (name, r) in zip(axes.flatten(), results.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        ConfusionMatrixDisplay(cm, display_labels=['Default', 'Repaid']).plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(name, fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, '02_confusion_matrices.png'), dpi=FIG_DPI); plt.close()
    bm = models[best_name]
    if hasattr(bm, 'feature_importances_'):
        imp = bm.feature_importances_
    elif hasattr(bm, 'coef_'):
        imp = np.abs(bm.coef_[0])
    else:
        imp = None
    if imp is not None:
        fn = comp.index.tolist() if len(comp.index) == len(imp) else [str(i) for i in range(len(imp))]
        fi = pd.Series(imp, index=range(len(imp))).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        fi.tail(20).plot(kind='barh', ax=ax, color='#2196F3', edgecolor='white')
        ax.set_title(f'Top 20 Features - {best_name}', fontsize=15, fontweight='bold')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, '03_feature_importance.png'), dpi=FIG_DPI); plt.close()
    metrics_p = ['Accuracy', 'Precision', 'Recall (Repaid)', 'Recall (Default)', 'F1-Score', 'ROC-AUC']
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metrics_p)); w = 0.2
    for i, (mn, row) in enumerate(comp[metrics_p].iterrows()):
        bars = ax.bar(x + i * w, row.values, w, label=mn, color=COLORS[i], edgecolor='white')
        for b, v in zip(bars, row.values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x + w * 1.5); ax.set_xticklabels(metrics_p, fontsize=11)
    ax.set_ylabel('Score'); ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0, 1.12)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, '04_model_comparison.png'), dpi=FIG_DPI); plt.close()
    try:
        df_raw = pd.read_excel(raw_data_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df_raw['LoanStatus'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color=['#E91E63', '#4CAF50'], edgecolor='white')
        axes[0].set_xticklabels(['Default(0)', 'Repaid(1)'], rotation=0)
        axes[0].set_title('Loan Status Distribution', fontweight='bold')
        df_raw['ApprovalDate'] = pd.to_datetime(df_raw['ApprovalDate'], errors='coerce')
        yearly = df_raw.groupby(df_raw['ApprovalDate'].dt.year)['LoanStatus'].apply(lambda x: (x == 0).mean())
        yearly.plot(kind='bar', ax=axes[1], color='#FF9800', edgecolor='white')
        axes[1].set_title('Default Rate by Year', fontweight='bold')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, '05_class_distribution.png'), dpi=FIG_DPI); plt.close()
    except Exception as e:
        logger.warning(f"Class dist plot failed: {e}")
    logger.info("All plots saved")

def save_artifacts(models, data, cv_scores, comp, best_name, output_dir):
    logger.info("PHASE 7: SAVING ARTIFACTS")
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        fname = f"credit_model_{name.replace(' ', '_')}.pkl"
        with open(os.path.join(output_dir, fname), 'wb') as f:
            pickle.dump(model, f)
    with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(models[best_name], f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(data['scaler'], f)
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(data['feature_names'], f, indent=2)
    metadata = {
        'version': MODEL_VERSION, 'trained_at': datetime.now().isoformat(),
        'best_model': best_name, 'split_method': data['split_method'],
        'n_features': len(data['feature_names']), 'feature_names': data['feature_names'],
        'train_samples': int(data['X_train'].shape[0]), 'test_samples': int(data['X_test'].shape[0]),
        'cv_folds': CV_FOLDS,
        'cv_scores': {n: {'mean_auc': float(s.mean()), 'std_auc': float(s.std()), 'fold_scores': s.tolist()} for n, s in cv_scores.items()},
        'test_metrics': {n: {k: float(v) for k, v in row.items()} for n, row in comp.iterrows()},
        'sklearn_version': __import__('sklearn').__version__,
        'xgboost_version': __import__('xgboost').__version__,
        'python_version': sys.version.split()[0],
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("All artifacts saved")

def main():
    parser = argparse.ArgumentParser(description='EBO-SACCO Training v2.0')
    parser.add_argument('--data', type=str, default=DATA_FILE)
    args = parser.parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"EBO-SACCO v{MODEL_VERSION} Training")
    df = load_data(args.data)
    df = preprocess(df)
    data = split_data(df)
    models, cv_scores = train_models(data)
    results, comp, best_name = evaluate_models(models, data, MODELS_DIR)
    generate_plots(results, comp, best_name, models, data['y_test'], args.data, PLOTS_DIR)
    save_artifacts(models, data, cv_scores, comp, best_name, MODELS_DIR)
    logger.info(f"COMPLETE: Best={best_name} AUC={comp.loc[best_name, 'ROC-AUC']:.4f}")

if __name__ == '__main__':
    main()
