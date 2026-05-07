"""
Supply Chain Lead Time Variance Analysis
Analyzes 3 years of purchase order data across 8 suppliers and 5 product categories.
Quantifies variance, scores supplier reliability, and identifies delay root causes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="darkgrid")

# ── Generate synthetic PO dataset ────────────────────────────────────────────
SUPPLIERS = {
    "SupplierA": {"base_lt": 14, "variance": 2.0, "region": "Domestic"},
    "SupplierB": {"base_lt": 21, "variance": 5.5, "region": "Asia"},
    "SupplierC": {"base_lt": 10, "variance": 1.5, "region": "Domestic"},
    "SupplierD": {"base_lt": 30, "variance": 9.0, "region": "Europe"},
    "SupplierE": {"base_lt": 18, "variance": 4.0, "region": "Asia"},
    "SupplierF": {"base_lt": 25, "variance": 7.5, "region": "South America"},
    "SupplierG": {"base_lt": 12, "variance": 2.5, "region": "Domestic"},
    "SupplierH": {"base_lt": 35, "variance": 12.0, "region": "Asia"},
}
CATEGORIES = ["Raw Materials", "Packaging", "Electronics", "Chemicals", "Spare Parts"]
DELAY_REASONS = ["Port Congestion", "Supplier Capacity", "Quality Hold",
                 "Customs Delay", "Weather", "On Time"]

records = []
order_date = pd.date_range("2022-01-01", "2024-12-31", freq="B")

for _ in range(2000):
    supplier = np.random.choice(list(SUPPLIERS.keys()))
    info = SUPPLIERS[supplier]
    promised_lt = info["base_lt"] + np.random.randint(-2, 3)
    actual_lt = max(1, round(np.random.normal(info["base_lt"], info["variance"])))
    variance_days = actual_lt - promised_lt

    # Delay reason weighted by variance magnitude
    if variance_days > 5:
        reason = np.random.choice(DELAY_REASONS[:-1], p=[0.30, 0.25, 0.20, 0.15, 0.10])
    elif variance_days > 0:
        reason = np.random.choice(DELAY_REASONS[:-1], p=[0.20, 0.20, 0.20, 0.20, 0.20])
    else:
        reason = "On Time"

    records.append({
        "order_date": np.random.choice(order_date),
        "supplier": supplier,
        "region": info["region"],
        "category": np.random.choice(CATEGORIES),
        "promised_lead_time": promised_lt,
        "actual_lead_time": actual_lt,
        "variance_days": variance_days,
        "unit_cost": round(np.random.uniform(10, 500), 2),
        "order_qty": np.random.randint(50, 2000),
        "delay_reason": reason,
    })

df = pd.DataFrame(records)
df["order_date"] = pd.to_datetime(df["order_date"])
df["on_time"] = (df["variance_days"] <= 0).astype(int)
df["holding_cost"] = df["variance_days"].clip(lower=0) * df["unit_cost"] * df["order_qty"] * 0.0003
df["quarter"] = df["order_date"].dt.to_period("Q")

# ── EDA ──────────────────────────────────────────────────────────────────────
print("=== Lead Time Variance Summary (days) ===")
print(df.groupby("supplier")["variance_days"].describe().round(2))

print("\n=== On-Time Delivery Rate by Supplier ===")
otd = df.groupby("supplier")["on_time"].mean().sort_values(ascending=False)
print((otd * 100).round(1).astype(str) + "%")

print(f"\n=== Total Estimated Holding Cost from Late Orders ===")
print(f"${df['holding_cost'].sum():,.0f}")

# ── Supplier Reliability Scorecard ───────────────────────────────────────────
scorecard = df.groupby("supplier").agg(
    otd_rate=("on_time", "mean"),
    avg_variance=("variance_days", "mean"),
    std_variance=("variance_days", "std"),
    total_orders=("supplier", "count"),
    holding_cost=("holding_cost", "sum"),
).reset_index()

# Score: 50% OTD + 30% low variance + 20% low std
scorecard["score"] = (
    scorecard["otd_rate"] * 50
    + (1 - scorecard["avg_variance"].clip(lower=0) / scorecard["avg_variance"].max()) * 30
    + (1 - scorecard["std_variance"] / scorecard["std_variance"].max()) * 20
).round(1)

print("\n=== Supplier Reliability Scorecard ===")
print(scorecard[["supplier", "otd_rate", "avg_variance", "score"]]
      .sort_values("score", ascending=False).to_string(index=False))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Supply Chain Lead Time Variance Analysis", fontsize=16, fontweight="bold")

# 1. Variance distribution by supplier (box plot)
order = df.groupby("supplier")["variance_days"].median().sort_values().index
sns.boxplot(data=df, x="supplier", y="variance_days", order=order,
            ax=axes[0, 0], palette="coolwarm", linewidth=1.2)
axes[0, 0].axhline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.7)
axes[0, 0].set_title("Lead Time Variance Distribution by Supplier")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Variance (days)")
axes[0, 0].tick_params(axis="x", rotation=30)

# 2. On-time delivery rate by supplier
colors = ["#69f0ae" if v >= 0.75 else "#ef9a9a" for v in otd.values]
axes[0, 1].barh(otd.index, otd.values * 100, color=colors)
axes[0, 1].axvline(75, color="white", linewidth=1.2, linestyle="--", alpha=0.7)
axes[0, 1].set_title("On-Time Delivery Rate by Supplier")
axes[0, 1].set_xlabel("OTD Rate (%)")
green_patch = mpatches.Patch(color="#69f0ae", label="≥75% target")
red_patch = mpatches.Patch(color="#ef9a9a", label="<75% target")
axes[0, 1].legend(handles=[green_patch, red_patch], fontsize=8)

# 3. Delay reason breakdown (late orders only)
late = df[df["variance_days"] > 0]
reason_counts = late["delay_reason"].value_counts()
axes[1, 0].pie(reason_counts, labels=reason_counts.index, autopct="%1.1f%%",
               colors=sns.color_palette("tab10", len(reason_counts)), startangle=90,
               textprops={"fontsize": 8})
axes[1, 0].set_title("Root Cause Breakdown (Late Orders Only)")

# 4. Quarterly variance trend (avg days late, by region)
quarterly = df.groupby(["quarter", "region"])["variance_days"].mean().reset_index()
quarterly["quarter_dt"] = quarterly["quarter"].dt.to_timestamp()
for region, color in zip(df["region"].unique(), sns.color_palette("tab10", df["region"].nunique())):
    subset = quarterly[quarterly["region"] == region]
    axes[1, 1].plot(subset["quarter_dt"], subset["variance_days"],
                    marker="o", label=region, color=color, linewidth=1.8, markersize=5)
axes[1, 1].axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.6)
axes[1, 1].set_title("Quarterly Avg Variance by Region")
axes[1, 1].set_ylabel("Avg Variance (days)")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("supply_chain_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to supply_chain_analysis.png")
