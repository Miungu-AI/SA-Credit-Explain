from src.accountability.audit_logger import log_decision, verify_chain

def test_hash_chain_integrity():
    log_decision(
        application_id="TEST-001", model_name="xgboost", model_version="0.1.0",
        application_input={"x": 1}, default_probability=0.42,
        decision="MANUAL_REVIEW", explanation={"test": True},
    )
    result = verify_chain()
    assert result["valid"] is True
    assert result["records"] >= 1
