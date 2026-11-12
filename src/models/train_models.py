"""
Train credit default models: Logistic Regression, Random Forest, XGBoost.
Protected attributes are EXCLUDED from model features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb

RANDOM_SEED = 42
DATA_PATH = Path("data/synthetic/sa_credit_synthetic.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Features that ARE allowed as model inputs (no protected attributes)
NUMERIC_FEATURES = [
    "monthly_income_zar",
    "debt_to_income_ratio",
    "existing_credit_obligations_zar",
    "repayment_history_score",
    "credit_utilisation_rate",
    "loan_amount_zar",
    "loan_duration_months",
]
CATEGORICAL_FEATURES = ["employment_status"]
TARGET = "default"
PROTECTED_ATTRIBUTES = ["gender", "age_category", "geographic_category", "income_category"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, df


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model_name": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def train_all():
    X_train, X_test, y_train, y_test, df = load_and_split()
    preprocessor = build_preprocessor()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_SEED,
            class_weight="balanced", n_jobs=-1
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_SEED, eval_metric="logloss",
            use_label_encoder=False
        ),
    }

    results = {}
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        # 5-fold cross-validation ROC-AUC on training set
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc"
        )
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(name, pipe, X_test, y_test)
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_roc_auc_std"] = round(cv_scores.std(), 4)
        results[name] = metrics

        # Save model pipeline
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipe, model_path)
        pipelines[name] = pipe
        print(f"[{name}] ROC-AUC: {metrics['roc_auc']} | CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save metrics
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save feature list for reproducibility
    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump({"features": FEATURES, "protected_attributes": PROTECTED_ATTRIBUTES}, f, indent=2)

    return pipelines, results, X_train, X_test, y_train, y_test, df


if __name__ == "__main__":
    train_all()
    print("\nModels and metrics saved to models/")
