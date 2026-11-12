"""
Regulatory governance mapping.
Distinguishes binding SA law from international guidance.
Statuses are INTERNAL PROTOTYPE ASSESSMENTS, not legal certifications.
"""

CONTROLS = [
    {
        "id": "TR-01",
        "domain": "Transparency",
        "control": "Model inputs and decision logic are documented and visible.",
        "sa_law": "POPIA s.18 (notification), s.71 (automated decisions); NCA s.63 (disclosure of reasons).",
        "international": "OECD AI Principle 1.3; NIST AI RMF GOVERN 1.1; EU AI Act Art.13 (comparative).",
        "status": "COMPLIANT",
        "evidence": "Model feature list, README, and explainability engine documentation.",
        "risk": "LOW",
        "action": "Maintain documentation as models evolve.",
    },
    {
        "id": "EX-01",
        "domain": "Explainability",
        "control": "Individual decisions are accompanied by plain-language explanations.",
        "sa_law": "POPIA s.71 (right to meaningful information about automated decisions); NCA s.63.",
        "international": "OECD 1.3; NIST AI RMF MAP 2.3; ISO/IEC 42001 A.6.1.2.",
        "status": "COMPLIANT",
        "evidence": "SHAP-based plain-language generator with limitations disclosure.",
        "risk": "LOW",
        "action": "User-test explanations with diverse literacy levels.",
    },
    {
        "id": "FA-01",
        "domain": "Fairness",
        "control": "Systematic group-level fairness auditing is performed.",
        "sa_law": "Constitution s.9 (equality); NCA s.61 (unlawful discrimination); POPIA s.71.",
        "international": "NIST AI RMF MEASURE 2.6; ISO/IEC 23894 6.4; EU AI Act Art.10 (comparative).",
        "status": "PARTIALLY_IMPLEMENTED",
        "evidence": "Fairlearn audit across 4 protected attributes; synthetic bias detected.",
        "risk": "MEDIUM",
        "action": "Implement mitigation strategies (reweighing, threshold optimisation) and re-audit.",
    },
    {
        "id": "AC-01",
        "domain": "Accountability",
        "control": "Decisions are logged in an immutable-style, verifiable audit trail.",
        "sa_law": "POPIA s.19 (security safeguards); NCA s.63; SARB/PA outsourcing and risk directives.",
        "international": "NIST AI RMF GOVERN 4.1; ISO/IEC 42001 A.7.5.",
        "status": "COMPLIANT",
        "evidence": "SHA-256 hash-chained SQLite audit log with integrity verification function.",
        "risk": "LOW",
        "action": "Consider WORM storage or external notarisation for production.",
    },
    {
        "id": "HO-01",
        "domain": "Human Oversight",
        "control": "Borderline and flagged decisions are routed to human reviewers.",
        "sa_law": "POPIA s.71(1) (right to human intervention); NCA s.63.",
        "international": "OECD 1.5; NIST AI RMF MANAGE 3.2; EU AI Act Art.14 (comparative).",
        "status": "PARTIALLY_IMPLEMENTED",
        "evidence": "Review queue and override recording implemented. Reviewer training not yet simulated.",
        "risk": "MEDIUM",
        "action": "Define reviewer competency requirements and escalation SLAs.",
    },
    {
        "id": "DP-01",
        "domain": "Data Protection",
        "control": "Sensitive attributes are not model inputs; PII is minimised in logs.",
        "sa_law": "POPIA s.9, s.10, s.11, s.19, s.22 (breach notification).",
        "international": "ISO/IEC 42001 A.8.2; NIST AI RMF GOVERN 1.3.",
        "status": "COMPLIANT",
        "evidence": "Protected attributes excluded from features; input hashed in audit log.",
        "risk": "LOW",
        "action": "Conduct a formal DPIA before any real-data pilot.",
    },
    {
        "id": "MR-01",
        "domain": "Model Risk Management",
        "control": "Model performance is monitored over time (PSI, drift, calibration).",
        "sa_law": "SARB/PA model risk guidance; FSCA conduct standards.",
        "international": "NIST AI RMF MEASURE 2.4; ISO/IEC 23894 6.5.",
        "status": "PARTIALLY_IMPLEMENTED",
        "evidence": "Monitoring module implemented with PSI and drift logic; production integration pending.",
        "risk": "MEDIUM",
        "action": "Define retraining triggers and model validation cadence.",
    },
    {
        "id": "AU-01",
        "domain": "Auditability",
        "control": "Models are versioned and experiments are reproducible.",
        "sa_law": "POPIA s.19; NCA record-keeping requirements.",
        "international": "ISO/IEC 42001 A.7.5; NIST AI RMF GOVERN 1.6.",
        "status": "COMPLIANT",
        "evidence": "Fixed random seeds, joblib model versions, MLflow tracking.",
        "risk": "LOW",
        "action": "Maintain model cards for each production candidate.",
    },
]


def get_governance_summary():
    total = len(CONTROLS)
    compliant = sum(1 for c in CONTROLS if c["status"] == "COMPLIANT")
    partial = sum(1 for c in CONTROLS if c["status"] == "PARTIALLY_IMPLEMENTED")
    gaps = sum(1 for c in CONTROLS if c["status"] == "GAP_IDENTIFIED")
    return {
        "total_controls": total,
        "compliant": compliant,
        "partially_implemented": partial,
        "gaps": gaps,
        "disclaimer": "Internal prototype assessment only. Not a formal legal compliance certification.",
        "controls": CONTROLS,
    }
