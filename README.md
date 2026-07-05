# EBO-SACCO Credit Scoring System v2.0

A supervised machine learning framework for loan repayment prediction, developed for EBO-SACCO microfinance institution in Western Uganda. Built as part of an MSc IT research project at Victoria University, Kampala.

## Architecture

The system uses a decoupled architecture with three interfaces:

- **Training Pipeline** (`train.py`) — preprocesses data, engineers features, trains 4 ML models with cross-validation, generates evaluation plots, and saves versioned model artifacts
- **Flask REST API** (`api.py`) — serves predictions with SHAP explainability and business rule guardrails
- **Streamlit App** (`app.py`) — alternative interactive frontend
- **Web Frontend** (`webapp/`) — glassmorphism HTML/CSS/JS interface that calls the Flask API

## Models

Four algorithms are trained and compared (as specified in the research methodology):

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost (uses `scale_pos_weight` instead of SMOTE)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (requires dataset in project root)
python train.py

# 3. Evaluate models (optional — generates SHAP plots)
python eval_model.py

# 4a. Start Flask API (for webapp)
python api.py
# Then open webapp/index.html in a browser

# 4b. Or start Streamlit app
streamlit run app.py
```

## Project Structure

```
CreditScoringProject/
├── config.py                  # Centralized configuration
├── train.py                   # Training pipeline (v2.0)
├── eval_model.py              # Evaluation with McNemar's test + SHAP
├── api.py                     # Flask API with SHAP explainability
├── app.py                     # Streamlit frontend
├── requirements.txt           # Frozen dependencies
├── Loan History_Dataset (1).xlsx  # Dataset (17,000 records)
├── models/                    # Trained model artifacts
│   ├── credit_model_*.pkl     # Individual model files
│   ├── best_model.pkl         # Best performing model
│   ├── scaler.pkl             # StandardScaler
│   ├── feature_names.json     # Feature list
│   ├── model_metadata.json    # Version, metrics, training info
│   └── model_comparison.csv   # Performance comparison table
├── plots/                     # Generated visualizations
│   ├── 01_roc_curves.png
│   ├── 02_confusion_matrices.png
│   ├── 03_feature_importance.png
│   ├── 04_model_comparison.png
│   ├── 05_class_distribution.png
│   ├── 06_shap_summary.png
│   └── 07_shap_bar.png
├── logs/                      # Training and API logs
├── webapp/                    # HTML/JS frontend
│   ├── index.html
│   ├── script.js
│   └── styles.css
└── README.md
```

## Key Improvem