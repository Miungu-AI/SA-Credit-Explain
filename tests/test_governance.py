from src.governance.regulatory_mapping import get_governance_summary

def test_governance_summary():
    gov = get_governance_summary()
    assert gov["total_controls"] == 8
    assert "disclaimer" in gov
    for c in gov["controls"]:
        assert c["status"] in ["COMPLIANT", "PARTIALLY_IMPLEMENTED", "GAP_IDENTIFIED"]
