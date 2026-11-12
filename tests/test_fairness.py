from src.fairness.fairness_audit import run_full_audit

def test_audit_returns_all_attributes():
    results = run_full_audit()
    assert "gender" in results
    assert "geographic_category" in results
    for attr, data in results.items():
        assert "group_metrics" in data
        assert len(data["group_metrics"]) >= 2
        assert 0 <= data["disparate_impact_ratio"] <= 1.0
