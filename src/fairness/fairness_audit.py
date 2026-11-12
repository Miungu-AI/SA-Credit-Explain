"""
Fairness auditing module using Fairlearn.
Protected attributes are used ONLY for auditing, never as model inputs.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.metrics import confusion_matrix

MODEL_DIR = Path("models")

# Illustrative decision threshold for binary audit (P(default) >= 0.55 -> decline)
AUDIT_THRESHOLD = 0.55


def load_predictions(model_name="xgboost"):
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    df = pd.read_csv("data/synthetic/sa_credit_synthetic.csv")
    features = joblib.load(MODEL_DIR / "feature_list.json")["features"]
    X = df[features]
    y = df["default"]
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= AUDIT_THRESHOLD).astype(int)  # 1 = declined
    return df, y, y_pred, y_proba


def audit_group(df, y_true, y_pred, protected_col):
    """Compute fairness metrics for a single protected attribute."""
    groups = df[protected_col].values
    unique_groups = sorted(df[protected_col].unique())

    results = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        results.append({
            "group": str(g),
            "n": int(mask.sum()),
            "selection_rate": round(float(yp.mean()), 4),
            "fpr": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
            "fnr": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),
            "tpr": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
            "accuracy": round(float((yt == yp).mean()), 4),
        })

    # Aggregate metrics
    metric_frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_true, y_pred=y_pred, sensitive_features=groups,
    )

    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=groups)

    # Disparate impact ratio (min selection rate / max selection rate)
    rates = [r["selection_rate"] for r in results]
    di_ratio = round(min(rates) / max(rates), 4) if max(rates) > 0 else 1.0

    return {
        "protected_attribute": protected_col,
        "group_metrics": results,
        "demographic_parity_difference": round(float(dp_diff), 4),
        "equalized_odds_difference": round(float(eo_diff), 4),
        "disparate_impact_ratio": di_ratio,
        "overall_selection_rate": round(float(y_pred.mean()), 4),
    }


def run_full_audit(model_name="xgboost"):
    df, y_true, y_pred, y_proba = load_predictions(model_name)
    protected_attributes = ["gender", "age_category", "geographic_category", "income_category"]
    audit_results = {}
    for attr in protected_attributes:
        audit_results[attr] = audit_group(df, y_true, y_pred, attr)
    return audit_results


if __name__ == "__main__":
    results = run_full_audit()
    import json
    with open(MODEL_DIR / "fairness_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Fairness audit complete. Saved to models/fairness_audit.json")
