"""
COVID-19 Trend Analysis
Explores simulated global pandemic data across 10 countries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(7)
sns.set_theme(style="darkgrid")

# ── Generate synthetic country-level data ────────────────────────────────────
countries = {
    "USA": {"pop": 331e6, "vax_rate": 0.68},
    "Brazil": {"pop": 213e6, "vax_rate": 0.58},
    "India": {"pop": 1380e6, "vax_rate": 0.44},
    "Germany": {"pop": 83e6, "vax_rate": 0.74},
    "UK": {"pop": 67e6, "vax_rate": 0.72},
    "France": {"pop": 67e6, "vax_rate": 0.70},
    "Japan": {"pop": 126e6, "vax_rate": 0.80},
    "Nigeria": {"pop": 206e6, "vax_rate": 0.04},
    "Australia": {"pop": 25e6, "vax_rate": 0.84},
    "Canada": {"pop": 38e6, "vax_rate": 0.82},
}

dates = pd.date_range("2020-03-01", "2022-12-31", freq="W")
records = []

for country, info in countries.items():
    cases_base = info["pop"] * np.random.uniform(0.002, 0.015)
    for i, date in enumerate(dates):
        wave_factor = 1 + 0.8 * np.sin(i / 15) + 0.4 * np.sin(i / 8)
        cases = max(0, int(cases_base * wave_factor * np.random.uniform(0.8, 1.2)))
        deaths = int(cases * np.random.uniform(0.008, 0.025) * (1 - info["vax_rate"] * 0.6))
        records.append({
            "date": date,
            "country": country,
            "new_cases": cases,
            "new_deaths": deaths,
            "vax_rate": info["vax_rate"],
            "population": info["pop"],
        })

df = pd.DataFrame(records)
df["cases_per_million"] = df["new_cases"] / (df["population"] / 1e6)
df["cfr"] = df["new_deaths"] / df["new_cases"].replace(0, np.nan)

# ── EDA ──────────────────────────────────────────────────────────────────────
print("=== Dataset Overview ===")
print(df.groupby("country")[["new_cases", "new_deaths"]].sum().sort_values("new_cases", ascending=False))

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("COVID-19 Global Trend Analysis", fontsize=16, fontweight="bold")

# 1. Weekly cases — select 4 countries
focus = ["USA", "India", "Germany", "Japan"]
palette = sns.color_palette("tab10", len(focus))
for country, color in zip(focus, palette):
    subset = df[df["country"] == country]
    axes[0, 0].plot(subset["date"], subset["cases_per_million"],
                    label=country, color=color, linewidth=1.8)
axes[0, 0].set_title("Weekly Cases per Million")
axes[0, 0].set_ylabel("Cases / 1M population")
axes[0, 0].legend(fontsize=8)

# 2. Total deaths by country
total_deaths = df.groupby("country")["new_deaths"].sum().sort_values(ascending=False)
axes[0, 1].barh(total_deaths.index, total_deaths.values / 1000,
                color=sns.color_palette("Reds_r", len(total_deaths)))
axes[0, 1].set_title("Estimated Total Deaths by Country")
axes[0, 1].set_xlabel("Deaths (thousands)")

# 3. Vaccination rate vs CFR scatter
summary = df.groupby("country").agg(
    vax_rate=("vax_rate", "first"),
    cfr=("cfr", "mean")
).reset_index()
axes[1, 0].scatter(summary["vax_rate"] * 100, summary["cfr"] * 100,
                   s=120, color="#4fc3f7", edgecolors="white", linewidths=0.8)
for _, row in summary.iterrows():
    axes[1, 0].annotate(row["country"], (row["vax_rate"] * 100, row["cfr"] * 100),
                        fontsize=7, xytext=(4, 2), textcoords="offset points")
axes[1, 0].set_title("Vaccination Rate vs Case Fatality Rate")
axes[1, 0].set_xlabel("Vaccination Rate (%)")
axes[1, 0].set_ylabel("Avg CFR (%)")

# 4. Correlation heatmap
corr_df = df.groupby("country").agg(
    total_cases=("new_cases", "sum"),
    total_deaths=("new_deaths", "sum"),
    vax_rate=("vax_rate", "first"),
    cfr=("cfr", "mean"),
    cases_pm=("cases_per_million", "mean"),
)
sns.heatmap(corr_df.corr(), ax=axes[1, 1], annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True)
axes[1, 1].set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("covid_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to covid_analysis.png")
