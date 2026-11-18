import re
import numpy as np
import pandas as pd

RAW = "Telco_customer_churn.xlsx"
OUT_DATA = "telco_clean.csv"
OUT_REPORT = "cleaning_report.txt"

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ---------------------------------------------------------------- 1. load
df = pd.read_excel(RAW, sheet_name="Telco_Churn")
log(f"[LOAD] raw shape = {df.shape}")

# ------------------------------------------------- 2. drop unusable columns
# 2a. ID column: unique per row, no signal
id_cols = ["CustomerID"]

# 2b. constant columns: zero variance, carry no information
const_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 1]
const_cols = [c for c in const_cols if c not in id_cols]

# 2c. LEAKAGE columns -- measured only AFTER / AT churn, must be removed
#     Churn Score : IBM pre-computed churn propensity score -> direct target leak
#     Churn Reason: only filled for customers who already churned
#     Churn Label : string duplicate of Churn Value (kept as value, dropped as text)
#     CLTV        : customer lifetime value, computed from retention history
#                   (survivors get higher CLTV) -> leaks the outcome
leak_cols = ["Churn Score", "Churn Reason", "Churn Label", "CLTV"]

# 2d. geo columns: 1652 distinct Zips / coordinates, extremely high cardinality and
#     no business value for a 7043-row model -> excluded from the 19-predictor set
geo_cols = ["City", "Zip Code", "Lat Long", "Latitude", "Longitude"]

drop_cols = id_cols + const_cols + leak_cols + geo_cols
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

log("")
log("[DROP] id        : " + ", ".join(id_cols))
log("[DROP] constant  : " + ", ".join(const_cols))
log("[DROP] leakage   : " + ", ".join(leak_cols))
log("[DROP] geo       : " + ", ".join(geo_cols))
log(f"[DROP] remaining shape = {df.shape}")

# --------------------------------------------- 3. fix Total Charges (object)
# 11 rows contain a blank string instead of a number. All of them have
# Tenure Months == 0, i.e. brand-new customers -> Total Charges = 0 is the
# correct value, not an unknown. We convert and assert that assumption.
before_missing = pd.to_numeric(df["Total Charges"], errors="coerce").isna().sum()
df["Total Charges"] = (
    df["Total Charges"].astype(str).str.strip().replace({"": np.nan})
)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

blank_mask = df["Total Charges"].isna()
log("")
log(f"[TOTAL CHARGES] non-numeric/blank rows = {before_missing}")
log(f"[TOTAL CHARGES] their Tenure Months values = "
    f"{sorted(df.loc[blank_mask, 'Tenure Months'].unique().tolist())}")
assert (df.loc[blank_mask, "Tenure Months"] == 0).all(), \
    "blank TotalCharges found on a non-zero-tenure customer -> needs review"

df.loc[blank_mask, "Total Charges"] = 0.0
log(f"[TOTAL CHARGES] imputed {before_missing} blanks as 0.0 "
    f"(zero-tenure customers, verifiable assumption)")
log(f"[TOTAL CHARGES] dtype now = {df['Total Charges'].dtype}, "
    f"missing = {df['Total Charges'].isna().sum()}")

# ------------------------------------- 4. collapse redundant category levels
# "No internet service" / "No phone service" both mean the same thing as "No":
# the add-on is not subscribed. Collapsing the 3-level variables to binary
# avoids duplicate dummy columns and keeps the encoding stable across models.
three_level = [
    "Multiple Lines", "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]
for c in three_level:
    n_before = df[c].nunique()
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    log(f"[COLLAPSE] {c:18s} levels {n_before} -> {df[c].nunique()}  "
        f"{df[c].value_counts().to_dict()}")

# ------------------------------------------------------ 5. binary encoding
# Yes/No -> 1/0 for the plain binary flags. Gender -> 1/0 (Male=1).
binary_map = {"No": 0, "Yes": 1}
for c in ["Senior Citizen", "Partner", "Dependents", "Phone Service",
          "Multiple Lines", "Online Security", "Online Backup",
          "Device Protection", "Tech Support", "Streaming TV",
          "Streaming Movies", "Paperless Billing"]:
    df[c] = df[c].map(binary_map).astype("int8")

df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype("int8")
log("")
log("[ENCODE] binary Yes/No -> 1/0 (12 flags) + Gender Female=0/Male=1")

# Target: keep numeric column, rename for clarity
df = df.rename(columns={"Churn Value": "Churn"})
df["Churn"] = df["Churn"].astype("int8")
log(f"[TARGET] Churn: 1 = {int(df['Churn'].sum())} "
    f"({df['Churn'].mean():.2%}), 0 = {int((1 - df['Churn']).sum())} "
    f"({1 - df['Churn'].mean():.2%})")

# --------------------------------------------------- 6. multi-level categories
# Contract / Internet Service / Payment Method stay as strings at this stage.
# build_features() encodes them, so every model owner produces identical dummies.
multilevel = ["Internet Service", "Contract", "Payment Method"]
log("[KEEP] multi-level (encoded later in build_features): "
    + ", ".join(multilevel))

# ------------------------------------------------------- 7. integrity checks
log("")
log("[CHECK] duplicates (full row) = " + str(df.duplicated().sum()))
log("[CHECK] total missing values = " + str(int(df.isna().sum().sum())))
log("[CHECK] dtypes:\n" + df.dtypes.to_string())

# domain sanity checks
assert df["Tenure Months"].between(0, 72).all(), "tenure out of range"
assert df["Monthly Charges"].between(0, 200).all(), "monthly charges out of range"
assert (df["Total Charges"] >= 0).all(), "negative total charges"

# Total Charges should be roughly Monthly * Tenure (not exact for mid-cycle changes)
expected = df["Monthly Charges"] * df["Tenure Months"]
gap = (df["Total Charges"] - expected).abs()
log("")
log("[CHECK] |TotalCharges - Monthly*Tenure| : median = "
    f"{gap.median():.2f}, 95th pct = {gap.quantile(0.95):.2f}, max = {gap.max():.2f}")
log("[CHECK] rows where gap > 100 : " + str(int((gap > 100).sum()))
    + "  (single-customer mid-cycle price changes; not a data error)")

# service-count aggregate (optional feature used in robustness checks)
service_cols = ["Phone Service", "Multiple Lines", "Online Security",
                "Online Backup", "Device Protection", "Tech Support",
                "Streaming TV", "Streaming Movies"]
df["ServiceCount"] = df[service_cols].sum(axis=1).astype("int8")
log("[FEATURE] ServiceCount = sum of the 8 add-on flags "
    f"(range {df['ServiceCount'].min()}-{df['ServiceCount'].max()})")

# ------------------------------------------------------------------ 8. save
df.to_csv(OUT_DATA, index=False)
log("")
log(f"[SAVE] {OUT_DATA}  shape = {df.shape}")
log("=" * 70)

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nreport -> {OUT_REPORT}")


# =====================================================================
# Shared feature builder -- all three model owners should call this
# so that LR / RF / XGBoost see exactly the same columns.
# =====================================================================
def build_features(df, drop_total_charges=False):
    """One-hot encode the three multi-level variables.

    Parameters
    ----------
    df : the cleaned dataframe returned by this script (telco_clean.csv)
    drop_total_charges : bool
        Used only for the robustness check required in the proposal
        ("whether results change when Total Charges is removed").

    Returns X (features), y (target).
    """
    X = df.drop(columns=["Churn"])
    if drop_total_charges:
        X = X.drop(columns=["Total Charges"])

    X = pd.get_dummies(
        X,
        columns=["Internet Service", "Contract", "Payment Method"],
        drop_first=True,          # avoid dummy traps for logistic regression
        dtype="int8",
    )
    y = df["Churn"].astype("int8")
    return X, y


if __name__ == "__main__":
    cleaned = pd.read_csv(OUT_DATA)
    X, y = build_features(cleaned)
    print(f"build_features -> X {X.shape}, y {y.shape}")
    print("columns:", list(X.columns))

