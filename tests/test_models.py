import pytest
import pandas as pd
from src.models.predict import predict_single

SAMPLE = {
    "monthly_income_zar": 25000,
    "debt_to_income_ratio": 0.35,
    "existing_credit_obligations_zar": 5000,
    "repayment_history_score": 0.75,
    "credit_utilisation_rate": 0.30,
    "loan_amount_zar": 100000,
    "loan_duration_months": 24,
    "employment_status": "Employed",
}

def test_prediction_returns_probability():
    result = predict_single(SAMPLE, model_name="xgboost")
    assert 0 <= result["default_probability"] <= 1
    assert result["illustrative_decision"] in ["APPROVED", "MANUAL_REVIEW", "DECLINED"]

def test_missing_field_raises():
    bad = {k: v for k, v in SAMPLE.items() if k != "monthly_income_zar"}
    with pytest.raises(ValueError):
        predict_single(bad)
