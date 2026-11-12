# SA-CreditExplain

An open-source framework for **Algorithmic Accountability and Explainability** in AI-driven credit decision-making in South African banking.

## ⚠️ Important Disclaimer
This is a **research prototype**. It uses **synthetic data**. It does **not** reproduce the lending policies of any South African bank. Compliance statuses are **internal prototype assessments**, not formal legal certifications.

## Modules
1. **AI Credit Decision Engine** – Logistic Regression, Random Forest, XGBoost.
2. **Explainable AI Engine** – SHAP, LIME, counterfactual explanations.
3. **Fairness & Bias Detection** – Fairlearn metrics across gender, age, income, geography.
4. **Accountability & Audit Trail** – Immutable-style logging with SHA-256 hash chaining.
5. **Human Oversight** – Review flags, overrides, escalation.
6. **Regulatory Governance Dashboard** – Mapping to POPIA, NCA, SARB/PA, FSCA, NCR, OECD, NIST, EU AI Act (comparative), ISO/IEC 42001 & 23894.
7. **Monitoring** – PSI, data drift, model drift.

## Quick Start
```bash
git clone https://github.com/Miungu-AI/SA-Credit-Explain.git
cd SA-CreditExplain
pip install -r requirements.txt
python -m src.data.generate_synthetic_data
python -m src.models.train_models
streamlit run app/main.py
