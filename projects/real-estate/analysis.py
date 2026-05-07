"""
Real Estate Market Analysis
Analyzes 5,000+ property listings across 6 US metro areas.
Identifies price drivers, appreciation trends, and builds a fair-value regression model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
sns.set_theme(style="darkgrid")

# ── Generate synthetic listings dataset ──────────────────────────────────────
METROS = {
    "Austin":       {"base_price": 480_000, "growth": 0.12, "sqft_avg": 2100},
    "Denver":       {"base_price": 560_000, "growth": 0.09, "sqft_avg": 1950},
    "Phoenix":      {"base_price": 395_000, "growth": 0.11, "sqft_avg": 2050},
    "Nashville":    {"base_price": 430_000, "growth": 0.10, "sqft_avg": 1900},
    "Charlotte":    {"base_price": 360_000, "growth": 0.08, "sqft_avg": 1850},
    "Indianapolis": {"base_price": 290_000, "growth": 0.06, "sqft_avg": 1750},
}
PROPERTY_TYPES = ["Single Family", "Condo", "Townhouse", "Multi-Family"]
YEARS = list(range(2019, 2025))

records = []
for _ in range(5200):
    metro = np.random.choice(list(METROS.keys()))
    info = METROS[metro]
    year = np.random.choice(YEARS)
    prop_type = np.random.choice(PROPERTY_TYPES, p=[0.55, 0.20, 0.15, 0.10])

    sqft = max(500, int(np.random.normal(info["sqft_avg"], 400)))
    beds = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.20, 0.40, 0.28, 0.07])
    baths = max(1, round(beds * np.random.uniform(0.5, 1.1)))
    age = np.random.randint(0, 60)
    garage = np.random.choice([0, 1, 2], p=[0.15, 0.45, 0.40])
    lot_sqft = max(0, int(np.random.normal(7500, 3000)))

    # Price model: base + appreciation + size + beds/baths + age penalty + garage
    price_mult = (1 + info["growth"]) ** (year - 2019)
    price = (
        info["base_price"] * price_mult
        + sqft * np.random.uniform(90, 130)
        + beds * 12_000
        + baths * 8_000
        - age * 1_200
        + garage * 18_000
        + lot_sqft * 2.5
        + np.random.normal(0, 25_000)
    )
    price = max(100_000, round(price, -3))

    records.append({
        "metro": metro,
        "year": year,
        "property_type": prop_type,
        "price": price,
        "sqft": sqft,
        "beds": beds,
        "baths": baths,
        "age_years": age,
        "garage_spaces": garage,
        "lot_sqft": lot_sqft,
        "price_per_sqft": round(price / sqft, 2),
    })

df = pd.DataFrame(records)

# ── EDA ──────────────────────────────────────────────────────────────────────
print("=== Dataset Overview ===")
print(df.describe().round(0))

print("\n=== Median Price by Metro ===")
print(df.groupby("metro")["price"].median().sort_values(ascending=False)
        .apply(lambda x: f"${x:,.0f}"))

print("\n=== Price/SqFt by Property Type ===")
print(df.groupby("property_type")["price_per_sqft"].mean().sort_values(ascending=False)
        .apply(lambda x: f"${x:.0f}"))

# ── Regression Model ─────────────────────────────────────────────────────────
df_enc = pd.get_dummies(df[["sqft", "beds", "baths", "age_years", "garage_spaces",
                              "lot_sqft", "metro", "property_type", "price"]],
                         columns=["metro", "property_type"], drop_first=True)
X = df_enc.drop("price", axis=1)
y = df_enc["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)
preds = model.predict(X_test_s)

r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"\n=== Regression Model Results ===")
print(f"R²  : {r2:.4f}")
print(f"MAE : ${mae:,.0f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Real Estate Market Analysis — 6 US Metros (2019–2024)",
             fontsize=16, fontweight="bold")

palette = sns.color_palette("tab10", df["metro"].nunique())
metro_colors = dict(zip(df["metro"].unique(), palette))

# 1. Median price appreciation over time by metro
yearly = df.groupby(["year", "metro"])["price"].median().reset_index()
for metro, color in metro_colors.items():
    subset = yearly[yearly["metro"] == metro]
    axes[0, 0].plot(subset["year"], subset["price"] / 1000,
                    marker="o", label=metro, color=color, linewidth=2, markersize=5)
axes[0, 0].set_title("Median Home Price Appreciation by Metro")
axes[0, 0].set_ylabel("Median Price ($K)")
axes[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
axes[0, 0].legend(fontsize=8)

# 2. Price per sqft distribution by metro (violin)
metro_order = df.groupby("metro")["price_per_sqft"].median().sort_values(ascending=False).index
sns.violinplot(data=df, x="metro", y="price_per_sqft", order=metro_order,
               ax=axes[0, 1], palette="tab10", inner="quartile", linewidth=1)
axes[0, 1].set_title("Price per SqFt Distribution by Metro")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Price / SqFt ($)")
axes[0, 1].tick_params(axis="x", rotation=20)

# 3. Actual vs Predicted price scatter
sample_idx = np.random.choice(len(y_test), 500, replace=False)
axes[1, 0].scatter(y_test.values[sample_idx] / 1000, preds[sample_idx] / 1000,
                   alpha=0.4, s=18, color="#ffca28", edgecolors="none")
lim_min = min(y_test.min(), preds.min()) / 1000
lim_max = max(y_test.max(), preds.max()) / 1000
axes[1, 0].plot([lim_min, lim_max], [lim_min, lim_max], "w--", linewidth=1.2, alpha=0.7)
axes[1, 0].set_title(f"Actual vs Predicted Price  (R² = {r2:.3f}, MAE = ${mae/1000:.0f}K)")
axes[1, 0].set_xlabel("Actual Price ($K)")
axes[1, 0].set_ylabel("Predicted Price ($K)")
axes[1, 0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
axes[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))

# 4. Feature importance (regression coefficients)
coef_df = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False).head(10)
axes[1, 1].barh(coef_df.index[::-1], coef_df.values[::-1],
                color=sns.color_palette("YlOrBr_r", 10))
axes[1, 1].set_title("Top 10 Price Drivers (|Coefficient|)")
axes[1, 1].set_xlabel("Absolute Coefficient")

plt.tight_layout()
plt.savefig("real_estate_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to real_estate_analysis.png")
