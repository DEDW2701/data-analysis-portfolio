"""
Retail Sales Analysis
Analyzes 2 years of simulated retail transaction data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="darkgrid", palette="muted")

# ── Generate synthetic dataset ──────────────────────────────────────────────
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Food & Beverage"]
regions = ["North", "South", "East", "West"]

records = []
for date in dates:
    for _ in range(np.random.randint(20, 80)):
        cat = np.random.choice(categories, p=[0.25, 0.20, 0.20, 0.15, 0.20])
        base_price = {"Electronics": 250, "Clothing": 60, "Home & Garden": 80,
                      "Sports": 90, "Food & Beverage": 30}[cat]
        records.append({
            "date": date,
            "category": cat,
            "region": np.random.choice(regions),
            "units": np.random.randint(1, 10),
            "unit_price": round(base_price * np.random.uniform(0.7, 1.4), 2),
        })

df = pd.DataFrame(records)
df["revenue"] = df["units"] * df["unit_price"]
df["month"] = df["date"].dt.to_period("M")
df["year"] = df["date"].dt.year

# ── EDA ──────────────────────────────────────────────────────────────────────
print("=== Dataset Overview ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nTotal Revenue: ${df['revenue'].sum():,.0f}")
print(f"Avg Daily Revenue: ${df.groupby('date')['revenue'].sum().mean():,.0f}")

# ── Plot 1: Monthly Revenue Trend ────────────────────────────────────────────
monthly = df.groupby(["month", "year"])["revenue"].sum().reset_index()
monthly["month_dt"] = monthly["month"].dt.to_timestamp()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Retail Sales Analysis — 2023–2024", fontsize=16, fontweight="bold")

for year, color in zip([2023, 2024], ["#4fc3f7", "#ef9a9a"]):
    subset = monthly[monthly["year"] == year]
    axes[0, 0].plot(subset["month_dt"], subset["revenue"] / 1000,
                    marker="o", label=str(year), color=color, linewidth=2)

axes[0, 0].set_title("Monthly Revenue Trend")
axes[0, 0].set_ylabel("Revenue ($K)")
axes[0, 0].legend()
axes[0, 0].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))

# ── Plot 2: Revenue by Category ──────────────────────────────────────────────
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
axes[0, 1].bar(cat_rev.index, cat_rev.values / 1000,
               color=sns.color_palette("muted", len(cat_rev)))
axes[0, 1].set_title("Revenue by Category")
axes[0, 1].set_ylabel("Revenue ($K)")
axes[0, 1].tick_params(axis="x", rotation=15)

# ── Plot 3: Revenue by Region ────────────────────────────────────────────────
region_rev = df.groupby("region")["revenue"].sum()
axes[1, 0].pie(region_rev, labels=region_rev.index, autopct="%1.1f%%",
               colors=sns.color_palette("pastel", len(region_rev)), startangle=90)
axes[1, 0].set_title("Revenue Split by Region")

# ── Plot 4: Category × Region Heatmap ───────────────────────────────────────
pivot = df.pivot_table(values="revenue", index="category",
                       columns="region", aggfunc="sum") / 1000
sns.heatmap(pivot, ax=axes[1, 1], annot=True, fmt=".0f",
            cmap="Blues", cbar_kws={"label": "$K"})
axes[1, 1].set_title("Revenue Heatmap (Category × Region)")

plt.tight_layout()
plt.savefig("retail_sales_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to retail_sales_analysis.png")
