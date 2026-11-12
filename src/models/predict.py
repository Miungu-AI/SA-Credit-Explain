"""
Load a trained model and generate predictions with illustrative lending thresholds.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

# Illustrative, configurable thresholds — NOT actual bank policy
APPROVE_THRESHOLD = 0.30   # P(default) < 0.30 -> approve
REVIEW_THRESHOLD = 0.55    # 0.30 <= P(default) < 0.55 -> manual review
# P(default) >= 0.55 -> decline


def load_model(model_name: str = "xgboost"):
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model {model_name} not found. Run train_models.py first.")
    return joblib.load(path)


def predict_single(application: dict, model_name: str = "xgboost"):
    model = load_model(model_name)
    df = pd.DataFrame([application])
    # Validate required features
    required = [
        "monthly_income_zar", "debt_to_income_ratio",
        "existing_credit_obligations_zar", "repayment_history_score",
        "credit_utilisation_rate", "loan_amount_zar",
        "loan_duration_months", "employment_status",
    ]
    missing = [f for f in required if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    proba = model.predict_proba(df)[0, 1]

    if proba < APPROVE_THRESHOLD:
        decision = "APPROVED"
    elif proba < REVIEW_THRESHOLD:
        decision = "MANUAL_REVIEW"
    else:
        decision = "DECLINED"

    return {
        "default_probability": round(float(proba), 4),
        "illustrative_decision": decision,
        "thresholds_used": {
            "approve_below": APPROVE_THRESHOLD,
            "review_below": REVIEW_THRESHOLD,
        },
        "disclaimer": "Illustrative decision only. Not actual bank lending policy.",
    }
