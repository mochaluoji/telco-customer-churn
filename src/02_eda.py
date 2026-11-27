import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv("data/processed/telco_clean.csv")
lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> figures/{name}")


CHURN_RATE = df["Churn"].mean()
log("=" * 70)
log(f"[EDA] n = {len(df)}, churn rate = {CHURN_RATE:.2%} "
    f"(imbalance ratio 1 : {1 / CHURN_RATE - 1:.1f})")
log("=" * 70)

# ------------------------------------------------------------- 1. univariate
num_cols = ["Tenure Months", "Monthly Charges", "Total Charges", "ServiceCount"]
log("\n[1] numeric summary")
log(df[num_cols].describe().round(2).to_string())

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, c in zip(axes, num_cols):
    sns.histplot(df[c], bins=30, ax=ax, color="#4C72B0")
    ax.set_title(c)
save(fig, "01_numeric_distributions.png")

# ------------------------------------------------- 2. churn vs numeric (split)
log("\n[2] numeric features by churn (mean)")
grp = df.groupby("Churn")[num_cols].mean().round(2)
grp.index = ["No churn", "Churn"]
log(grp.to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, c in zip(axes, ["Tenure Months", "Monthly Charges", "Total Charges"]):
    sns.boxplot(data=df, x="Churn", y=c, ax=ax,
                hue="Churn", palette={0: "#55A868", 1: "#C44E52"}, legend=False)
    ax.set_xticklabels(["No churn", "Churn"])
    ax.set_title(f"{c} by churn")
    ax.set_xlabel("")
save(fig, "02_numeric_vs_churn.png")

# --------------------------------------------------- 3. categorical churn rate
cat_cols = ["Contract", "Internet Service", "Payment Method",
            "Paperless Billing", "Tech Support", "Online Security"]
log("\n[3] churn rate by category")
for c in cat_cols:
    t = df.groupby(c)["Churn"].agg(n="size", churn_rate="mean")
    t["churn_rate"] = (t["churn_rate"] * 100).round(1)
    t = t.sort_values("churn_rate", ascending=False)
    log(f"\n  {c}")
    log(t.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for ax, c in zip(axes, ["Contract", "Internet Service", "Payment Method"]):
    r = df.groupby(c)["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=r.values, y=r.index, ax=ax,
                hue=r.index, palette="rocket", legend=False)
    ax.axvline(CHURN_RATE, ls="--", c="grey", lw=1)
    ax.set_title(f"Churn rate by {c}\n(dashed = overall {CHURN_RATE:.1%})")
    ax.set_xlabel("churn rate")
save(fig, "03_churn_rate_by_category.png")

# ------------------------------------------------------ 4. tenure + contract
log("\n[4] tenure buckets")
df["TenureBucket"] = pd.cut(df["Tenure Months"],
                            [-1, 6, 12, 24, 48, 72],
                            labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
tb = df.groupby("TenureBucket", observed=True)["Churn"].agg(n="size", rate="mean")
tb["rate"] = (tb["rate"] * 100).round(1)
log(tb.to_string())

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=tb.index, y=tb["rate"] / 100, ax=ax, color="#C44E52")
ax.axhline(CHURN_RATE, ls="--", c="grey")
ax.set_title(f"Churn rate by tenure bucket (overall {CHURN_RATE:.1%})")
ax.set_ylabel("churn rate")
save(fig, "04_churn_rate_by_tenure.png")

# contract × internet heatmap
pivot = df.pivot_table(index="Contract", columns="Internet Service",
                       values="Churn", aggfunc="mean")
log("\n  churn rate, Contract x Internet Service")
log((pivot * 100).round(1).to_string())
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar_kws={"label": "churn rate"})
ax.set_title("Churn rate: Contract x Internet Service")
save(fig, "05_heatmap_contract_internet.png")

# ------------------------------------------------------------ 5. correlations
num_for_corr = ["Tenure Months", "Monthly Charges", "Total Charges",
                "ServiceCount", "Churn"]
corr = df[num_for_corr].corr()
log("\n[5] Pearson correlation with churn")
log(corr["Churn"].round(3).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Correlation, numeric features + target")
save(fig, "06_correlation_matrix.png")

# ---------------------------------------------------------- 6. class balance
fig, ax = plt.subplots(figsize=(5, 4))
vc = df["Churn"].value_counts().sort_index()
sns.barplot(x=["No churn (0)", "Churn (1)"], y=vc.values, ax=ax,
            hue=["No churn (0)", "Churn (1)"],
            palette={"No churn (0)": "#55A868", "Churn (1)": "#C44E52"},
            legend=False)
for i, v in enumerate(vc.values):
    ax.text(i, v + 60, f"{v}\n({v / len(df):.1%})", ha="center", fontsize=10)
ax.set_ylim(0, vc.max() * 1.25)
ax.set_ylabel("customers")
ax.set_title("Class balance")
save(fig, "07_class_balance.png")

# ------------------------------------------------------------ 7. key findings
log("\n" + "=" * 70)
log("[KEY FINDINGS FOR SLIDES]")
log("=" * 70)

top_contract = df.groupby("Contract")["Churn"].mean().idxmax()
top_contract_rate = df.groupby("Contract")["Churn"].mean().max()
top_internet = df.groupby("Internet Service")["Churn"].mean().idxmax()
top_internet_rate = df.groupby("Internet Service")["Churn"].mean().max()
top_pay = df.groupby("Payment Method")["Churn"].mean().idxmax()
top_pay_rate = df.groupby("Payment Method")["Churn"].mean().max()

log(f"1. Imbalance: {CHURN_RATE:.1%} churn -> use AUC-ROC / F1, not accuracy; "
    f"stratified split required.")
log(f"2. Contract is the strongest categorical driver: "
    f"{top_contract} churn {top_contract_rate:.1%} vs "
    f"{df.groupby('Contract')['Churn'].mean().min():.1%} for the lowest group.")
log(f"3. Internet Service: {top_internet} churn {top_internet_rate:.1%}; "
    f"customers without internet churn only "
    f"{df.loc[df['Internet Service'] == 'No', 'Churn'].mean():.1%}.")
log(f"4. Payment: {top_pay} churn {top_pay_rate:.1%}, roughly 3x the "
    f"automatic-payment rate.")
log(f"5. Tenure: churned customers average "
    f"{df.loc[df['Churn'] == 1, 'Tenure Months'].mean():.1f} months vs "
    f"{df.loc[df['Churn'] == 0, 'Tenure Months'].mean():.1f} months for stayers; "
    f"churn rate in first 6 months = "
    f"{df.loc[df['Tenure Months'] <= 6, 'Churn'].mean():.1%}.")
log(f"6. Monthly Charges: churners pay "
    f"${df.loc[df['Churn'] == 1, 'Monthly Charges'].mean():.2f} vs "
    f"${df.loc[df['Churn'] == 0, 'Monthly Charges'].mean():.2f} "
    f"(high price + short tenure = high risk).")
log("7. Multicollinearity: Total Charges ~ Monthly Charges x Tenure "
    f"(r = {df['Total Charges'].corr(df['Monthly Charges'] * df['Tenure Months']):.2f}); "
    "flag for the robustness check / consider dropping Total Charges.")
log("8. Leakage removed before EDA: Churn Score, Churn Reason, CLTV, "
    "Churn Label. Churn Reason is post-hoc only and must never enter a model.")

with open("outputs/eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsummary -> eda_summary.txt")

