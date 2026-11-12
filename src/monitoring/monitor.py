"""
Model monitoring: performance, PSI, data drift, and fairness drift.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

MODEL_DIR = Path("models")


def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index."""
    def scale_range(input_array, min_val, max_val):
        input_array += -(np.min(input_array))
        input_array /= np.max(input_array) / (max_val - min_val)
        input_array += min_val
        return input_array

    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if e_perc == 0:
            e_perc = 0.0001
        if a_perc == 0:
            a_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    return np.sum([sub_psi(e, a) for e, a in zip(expected_percents, actual_percents)])


def monitor(model_name="xgboost", reference_frac=0.7):
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    df = pd.read_csv("data/synthetic/sa_credit_synthetic.csv")
    features = joblib.load(MODEL_DIR / "feature_list.json")["features"]

    split = int(len(df) * reference_frac)
    ref, cur = df.iloc[:split], df.iloc[split:]

    # Performance on current window
    y_true = cur["default"]
    y_proba = model.predict_proba(cur[features])[:, 1]
    y_pred = (y_proba >= 0.55).astype(int)

    performance = {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }

    # PSI for numeric features
    psi_scores = {}
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            psi_scores[col] = round(calculate_psi(ref[col].values, cur[col].values), 4)

    psi_alert = {k: v for k, v in psi_scores.items() if v > 0.25}

    return {
        "performance_current_window": performance,
        "psi_scores": psi_scores,
        "psi_alerts": psi_alert,
        "note": (
            "PSI > 0.25 indicates significant population shift. "
            "This is distinct from fairness deterioration: a model may maintain "
            "overall accuracy while becoming less fair for a specific group. "
            "Always run the fairness audit alongside performance monitoring."
        ),
    }
