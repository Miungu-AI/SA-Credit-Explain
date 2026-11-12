"""
FastAPI backend for SA-CreditExplain.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import joblib
from pathlib import Path

from src.models.predict import predict_single
from src.explainability.explainer import explain_local, generate_plain_language, generate_counterfactual, get_global_importance
from src.fairness.fairness_audit import run_full_audit
from src.accountability.audit_logger import log_decision, verify_chain, get_all_records
from src.accountability.review_manager import flag_for_review, record_review, get_review_queue
from src.governance.regulatory_mapping import get_governance_summary
from src.monitoring.monitor import monitor

app = FastAPI(title="SA-CreditExplain API", version="0.1.0")


class CreditApplication(BaseModel):
    monthly_income_zar: float = Field(..., gt=0)
    debt_to_income_ratio: float = Field(..., gt=0, le=1)
    existing_credit_obligations_zar: float = Field(..., ge=0)
    repayment_history_score: float = Field(..., ge=0, le=1)
    credit_utilisation_rate: float = Field(..., ge=0, le=1)
    loan_amount_zar: float = Field(..., gt=0)
    loan_duration_months: int = Field(..., gt=0)
    employment_status: str = Field(..., pattern="^(Employed|Self-Employed|Unemployed)$")
    model_name: str = "xgboost"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(app_input: CreditApplication):
    data = app_input.dict()
    model_name = data.pop("model_name")
    try:
        prediction = predict_single(data, model_name=model_name)
        explanation = explain_local(data, model_name=model_name)
        plain = generate_plain_language(explanation, prediction["illustrative_decision"], prediction["default_probability"])
        counterfactual = generate_counterfactual(data, model_name=model_name)
        review_flag = flag_for_review(prediction["default_probability"])

        app_id = f"API-{uuid.uuid4().hex[:8]}"
        record_hash = log_decision(
            application_id=app_id,
            model_name=model_name,
            model_version="0.1.0",
            application_input=data,
            default_probability=prediction["default_probability"],
            decision=prediction["illustrative_decision"],
            explanation={"top_factors": explanation["top_factors"], "plain_language": plain["plain_language"]},
            fairness_status="NOT_AUDITED",
            override_flag=0,
        )

        return {
            "application_id": app_id,
            "prediction": prediction,
            "explanation": explanation,
            "plain_language": plain,
            "counterfactual": counterfactual,
            "review_required": review_flag,
            "audit_record_hash": record_hash,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/fairness-audit")
def fairness_audit():
    return run_full_audit()


@app.get("/governance")
def governance():
    return get_governance_summary()


@app.get("/monitoring")
def monitoring():
    return monitor()


@app.get("/audit/verify")
def audit_verify():
    return verify_chain()


@app.get("/audit/records")
def audit_records():
    return get_all_records()


@app.get("/review/queue")
def review_queue():
    return get_review_queue()


@app.post("/review/{application_id}")
def submit_review(application_id: str, reviewer: str, outcome: str, reason: str = ""):
    return record_review(application_id, reviewer, outcome, reason)
