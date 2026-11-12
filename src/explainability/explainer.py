"""
Explainability engine: SHAP, LIME, counterfactuals, plain-language explanations.
"""
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
FEATURE_LABELS = {
    "monthly_income_zar": "Monthly income (ZAR)",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "existing_credit_obligations_zar": "Existing credit obligations (ZAR)",
    "repayment_history_score": "Repayment history score",
    "credit_utilisation_rate": "Credit utilisation rate",
    "loan_amount_zar": "Loan amount (ZAR)",
    "loan_duration_months": "Loan duration (months)",
    "employment_status": "Employment status",
}


def load_model_and_data(model_name="xgboost"):
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    df = pd.read_csv("data/synthetic/sa_credit_synthetic.csv")
    features = joblib.load(MODEL_DIR / "feature_list.json")["features"]
    X = df[features]
    return model, X, df


def get_shap_explainer(model, X_train, model_name="xgboost"):
    """Create a SHAP explainer appropriate for the model type."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    X_transformed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    if model_name == "logistic_regression":
        explainer = shap.LinearExplainer(classifier, X_transformed)
    elif model_name == "random_forest":
        explainer = shap.TreeExplainer(classifier)
    else:  # xgboost
        explainer = shap.TreeExplainer(classifier)

    return explainer, preprocessor, feature_names


def explain_local(application: dict, model_name="xgboost", top_k=5):
    model, X_train, _ = load_model_and_data(model_name)
    explainer, preprocessor, feature_names = get_shap_explainer(model, X_train, model_name)

    input_df = pd.DataFrame([application])
    input_transformed = preprocessor.transform(input_df)
    shap_values = explainer.shap_values(input_transformed)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # class 1 (default)
    else:
        sv = shap_values[0]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap", ascending=False).head(top_k)

    # Map transformed feature names back to original where possible
    contributions["friendly_name"] = contributions["feature"].apply(
        lambda x: FEATURE_LABELS.get(x.split("__")[-1], x)
    )

    positive = contributions[contributions["shap_value"] > 0]
    negative = contributions[contributions["shap_value"] < 0]

    return {
        "top_factors": contributions[["friendly_name", "shap_value"]].to_dict("records"),
        "positive_contributions": positive[["friendly_name", "shap_value"]].to_dict("records"),
        "negative_contributions": negative[["friendly_name", "shap_value"]].to_dict("records"),
        "base_value": float(explainer.expected_value if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1]),
    }


def generate_plain_language(explanation: dict, decision: str, probability: float):
    """Generate a factually grounded plain-language explanation."""
    factors = explanation["top_factors"]
    factor_descriptions = []

    for f in factors[:5]:
        direction = "increased" if f["shap_value"] > 0 else "decreased"
        factor_descriptions.append(
            f"{f['friendly_name']} {direction} the estimated default risk"
        )

    if decision == "DECLINED":
        intro = ("Your credit application was not approved because the model "
                 "identified a high estimated probability of repayment difficulty.")
    elif decision == "APPROVED":
        intro = ("Your credit application was approved because the model "
                 "estimated a low probability of repayment difficulty.")
    else:
        intro = ("Your credit application requires manual review because the "
                 "model's estimate falls in a borderline range.")

    body = "The main contributing factors were: " + "; ".join(factor_descriptions) + "."

    limitations = (
        "This explanation reflects the model's statistical associations, not causal relationships. "
        "Changing a single factor does not guarantee a different outcome. "
        "The decision is based on a synthetic-data prototype and is not an actual bank decision."
    )

    return {
        "plain_language": f"{intro} {body}",
        "limitations": limitations,
    }


def generate_counterfactual(application: dict, model_name="xgboost", desired_class=0):
    """
    Generate a simple counterfactual: find the minimal change to DTI that would
    flip the prediction to the desired class (0 = no default).
    """
    model, _, _ = load_model_and_data(model_name)
    original = application.copy()
    dti_original = original["debt_to_income_ratio"]

    # Search over DTI values
    for delta in np.arange(0.01, 0.50, 0.01):
        candidate = original.copy()
        candidate["debt_to_income_ratio"] = max(0.01, dti_original - delta)
        proba = model.predict_proba(pd.DataFrame([candidate]))[0, 1]
        pred = 1 if proba >= 0.5 else 0
        if pred == desired_class:
            return {
                "counterfactual": {
                    "field": "debt_to_income_ratio",
                    "original_value": round(float(dti_original), 4),
                    "suggested_value": round(float(candidate["debt_to_income_ratio"]), 4),
                    "change": round(float(-delta), 4),
                },
                "new_probability": round(float(proba), 4),
                "note": "This is an illustrative scenario, not a guarantee of approval.",
            }

    return {
        "counterfactual": None,
        "note": "No single DTI change within the searched range flipped the prediction.",
    }


def get_global_importance(model_name="xgboost"):
    """Global feature importance from SHAP."""
    model, X_train, _ = load_model_and_data(model_name)
    explainer, preprocessor, feature_names = get_shap_explainer(model, X_train, model_name)
    X_transformed = preprocessor.transform(X_train.sample(min(500, len(X_train)), random_state=42))
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    importance["friendly_name"] = importance["feature"].apply(
        lambda x: FEATURE_LABELS.get(x.split("__")[-1], x)
    )
    return importance.head(15).to_dict("records")
