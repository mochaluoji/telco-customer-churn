# Data Cleaning & EDA — Handoff Note

## 1. Files
| File | What it is |
|---|---|
| `01_data_cleaning.py` | Raw xlsx → clean csv. Run this first. |
| `02_eda.py` | Figures + `eda_summary.txt` (key numbers for slides). |
| `telco_clean.csv` | **The one table all three models use.** 7043 × 21. |
| `cleaning_report.txt` | Audit log: every drop, imputation and assertion. |
| `figures/*.png` | 7 EDA charts. |

## 2. Column contract (read before you encode)
- **Target:** `Churn` (1/0, 26.54% positive). Original `Churn Label` string column dropped.
- **Dropped as leakage** (never re-introduce):
  - `Churn Score` — pre-computed propensity score, direct target leak.
  - `Churn Reason` — only populated for customers who already churned.
  - `CLTV` — customer lifetime value derived from retention history.
  - `Churn Label` — text duplicate of the target.
- **Dropped as useless:** `CustomerID` (unique ID), `Count` / `Country` / `State` (constant), `City` / `Zip Code` / `Lat Long` / `Latitude` / `Longitude` (high-cardinality geo, excluded per proposal's 19-predictor set).
- **Binary 1/0 already done:** 12 Yes/No flags + `Gender` (Female=0, Male=1).
- **Still strings, one-hot needed:** `Internet Service` (3), `Contract` (3), `Payment Method` (4).
- **Collapsed:** "No internet service" / "No phone service" → `No`, so the 7 add-on variables are clean binary.
- **`Total Charges`:** 11 blank strings were all `Tenure Months == 0`; imputed to **0.0**, asserted in the script. Now float64, zero missing.

## 3. How to get your X / y (identical for everyone)
```python
import pandas as pd
from importlib.machinery import SourceFileLoader
dc = SourceFileLoader("dc", "01_data_cleaning.py").load_module()

df = pd.read_csv("telco_clean.csv")
X, y = dc.build_features(df)                        # 24 cols
X_r, y_r = dc.build_features(df, drop_total_charges=True)   # robustness check, 23 cols
```
One-hot uses `drop_first=True` → 24 features, no dummy trap (safe for logistic regression with an intercept; also fine for RF/XGBoost).

## 4. Things that matter for your own part
- **Stratify every split:** `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`.
- **Baseline floor:** majority-class accuracy = 73.46%, AUC = 0.5. Any model must beat both.
- **Standardize for LR only**, inside a `Pipeline` so scaling is fit on train folds only (no leakage through CV).
- **Class imbalance:** 26.5% is moderate, not extreme. Report F1 + AUC; if you try `class_weight='balanced'`, report both configurations rather than silently changing the metric.
- **`Total Charges` multicollinearity:** r = 1.00 with `Monthly Charges × Tenure`. Drop-in-column is intentional — use `drop_total_charges=True` for the robustness row in the results table.
- **SHAP:** use the cleaned feature names as-is so plots read consistently across models.

## 5. Numbers you can quote in slides
| Group | Churn rate |
|---|---|
| Month-to-month contract | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |
| Fiber optic internet | 41.9% |
| DSL | 19.0% |
| No internet | 7.4% |
| Electronic check payment | 45.3% |
| Automatic payment | ~16% |
| Tenure 0–6 months | 52.9% |
| Tenure 49–72 months | 9.5% |
| Month-to-month + Fiber optic | 54.6% |

Mean tenure: churners 18.0 months vs stayers 37.6. Mean monthly charge: churners $74.44 vs stayers $61.27.
