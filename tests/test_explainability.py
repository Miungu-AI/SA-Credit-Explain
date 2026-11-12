from src.explainability.explainer import explain_local, generate_plain_language
from tests.test_models import SAMPLE

def test_local_explanation_has_factors():
    exp = explain_local(SAMPLE)
    assert len(exp["top_factors"]) == 5
    assert "base_value" in exp

def test_plain_language_contains_limitations():
    exp = explain_local(SAMPLE)
    plain = generate_plain_language(exp, "DECLINED", 0.7)
    assert "not causal" in plain["limitations"].lower()
    assert len(plain["plain_language"]) > 0
