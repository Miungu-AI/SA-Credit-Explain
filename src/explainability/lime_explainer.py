import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path("models")

def lime_local(application: dict, model_name="xgboost"):
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    df = pd.read_csv("data/synthetic/sa_credit_synthetic.csv")
    features = joblib.load(MODEL_DIR / "feature_list.json")["features"]
    X_train = df[features]

    preprocessor = model.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_transformed,
        feature_names=feature_names,
        class_names=["No Default", "Default"],
        mode="classification",
        random_state=42,
    )

    input_transformed = preprocessor.transform(pd.DataFrame([application]))[0]
    exp = explainer.explain_instance(
        input_transformed,
        model.named_steps["classifier"].predict_proba,
        num_features=5,
    )

    return {
        "lime_explanation": [
            {"feature": f[0], "weight": round(f[1], 4)} for f in exp.as_list()
        ],
        "note": "LIME explanations are locally faithful approximations.",
    }
