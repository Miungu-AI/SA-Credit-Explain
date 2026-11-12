
**`LICENSE`** – Standard MIT Licence text.

---

## 3. Module 0: Synthetic South African Credit Dataset

**`src/data/generate_synthetic_data.py`**

This module generates a synthetic dataset with locally relevant variables and realistic correlations. It introduces **structural bias** (e.g., lower approval rates for certain geographic categories) to make fairness auditing meaningful. The data is **clearly labelled as synthetic**.

```python
"""
Generate a synthetic South African credit application dataset.
DISCLAIMER: This data is entirely synthetic. It does NOT represent actual
South African banking populations or statistics. It is designed for research
and demonstration purposes only.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SAMPLES = 5000

def generate_synthetic_data(n=N_SAMPLES):
    # --- Demographic variables (restricted audit variables, NOT model inputs) ---
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.48, 0.52])
    age = np.random.randint(18, 70, size=n)
    age_category = pd.cut(age, bins=[17, 25, 35, 50, 70],
                          labels=["18-25", "26-35", "36-50", "51-70"]).astype(str)

    # Geographic category: Urban (higher income potential), Peri-Urban, Rural
    # Structural disadvantage encoded here to simulate historical inequality
    geo_category = np.random.choice(
        ["Urban", "Peri-Urban", "Rural"], size=n, p=[0.55, 0.30, 0.15]
    )

    # Income is correlated with geography (simulating structural inequality)
    income_base = {"Urban": 25000, "Peri-Urban": 15000, "Rural": 8000}
    monthly_income_zar = np.array([
        max(1500, np.random.normal(income_base[g], income_base[g] * 0.4))
        for g in geo_category
    ])

    # Income category
    income_category = pd.cut(
        monthly_income_zar,
        bins=[0, 8000, 20000, 50000, np.inf],
        labels=["Low", "Lower-Middle", "Upper-Middle", "High"]
    ).astype(str)

    # --- Financial variables (model inputs) ---
    employment_status = np.random.choice(
        ["Employed", "Self-Employed", "Unemployed"], size=n, p=[0.70, 0.20, 0.10]
    )

    # Debt-to-income ratio: worse for unemployed/rural
    dti_base = np.where(
        employment_status == "Unemployed", 0.65,
        np.where(employment_status == "Self-Employed", 0.45, 0.35)
    )
    dti_base += np.where(geo_category == "Rural", 0.10, 0.0)  # structural penalty
    debt_to_income_ratio = np.clip(np.random.normal(dti_base, 0.15), 0.05, 0.95)

    existing_credit_obligations = np.clip(
        np.random.normal(5000, 3000, size=n), 0, 25000
    )

    # Repayment history: 0 = poor, 1 = excellent
    repayment_history = np.clip(
        np.random.beta(2, 2, size=n) - (dti_base - 0.4) * 0.3, 0, 1
    )

    credit_utilisation = np.clip(np.random.beta(2, 3, size=n), 0, 1)

    loan_amount = np.clip(
        monthly_income_zar * np.random.uniform(1, 8, size=n), 1000, 500000
    )
    loan_duration_months = np.random.choice([6, 12, 18, 24, 36, 48, 60], size=n)

    # --- Target: Default (1 = default) ---
    # Higher default risk for high DTI, poor repayment, high utilisation, low income
    logit = (
        -2.5
        + 3.5 * debt_to_income_ratio
        - 2.0 * repayment_history
        + 1.5 * credit_utilisation
        - 0.00003 * monthly_income_zar
        + 0.00001 * existing_credit_obligations
        + np.where(geo_category == "Rural", 0.6, 0.0)  # structural bias injected
        + np.random.normal(0, 0.8, size=n)  # noise
    )
    default_prob = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, default_prob)

    df = pd.DataFrame({
        "application_id": [f"SYN-{i:06d}" for i in range(n)],
        "gender": gender,
        "age": age,
        "age_category": age_category,
        "geographic_category": geo_category,
        "monthly_income_zar": np.round(monthly_income_zar, 2),
        "income_category": income_category,
        "employment_status": employment_status,
        "debt_to_income_ratio": np.round(debt_to_income_ratio, 4),
        "existing_credit_obligations_zar": np.round(existing_credit_obligations, 2),
        "repayment_history_score": np.round(repayment_history, 4),
        "credit_utilisation_rate": np.round(credit_utilisation, 4),
        "loan_amount_zar": np.round(loan_amount, 2),
        "loan_duration_months": loan_duration_months,
        "default": default,
    })
    return df


if __name__ == "__main__":
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(out_dir / "sa_credit_synthetic.csv", index=False)
    print(f"Generated {len(df)} synthetic applications.")
    print(f"Default rate: {df['default'].mean():.2%}")
    print("Data saved to data/synthetic/sa_credit_synthetic.csv")
    print("\n⚠️ DISCLAIMER: This data is SYNTHETIC and does not represent actual SA banking data.")
