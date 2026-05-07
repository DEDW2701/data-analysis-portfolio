"""
Stock Market EDA
Explores S&P 500 sector performance and volatility over 10 years.
Uses yfinance for real data (falls back to synthetic if unavailable).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
np.random.seed(99)

# ── Try real data, fall back to synthetic ───────────────────────────────────
TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Tech (NASDAQ)",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
}

try:
    import yfinance as yf
    raw = yf.download(list(TICKERS.keys()), start="2014-01-01", end="2024-01-01",
                      auto_adjust=True, progress=False)["Close"]
    raw.columns = [TICKERS[t] for t in raw.columns]
    print("Using real yfinance data.")
except Exception:
    print("yfinance unavailable — using synthetic data.")
    dates = pd.date_range("2014-01-01", "2024-01-01", freq="B")
    raw = pd.DataFrame(index=dates)
    starting = {"S&P 500": 1800, "Tech (NASDAQ)": 3500, "Energy": 80,
                "Financials": 22, "Healthcare": 60}
    for name, start in starting.items():
        returns = np.random.normal(0.0003, 0.012, len(dates))
        returns[np.random.choice(len(dates), 30, replace=False)] = np.random.uniform(-0.05, -0.02, 30)
        raw[name] = start * np.exp(np.cumsum(returns))

# ── Derived metrics ──────────────────────────────────────────────────────────
returns = raw.pct_change().dropna()
log_returns = np.log(raw / raw.shift(1)).dropna()

# Normalised performance (base = 100)
norm = (raw / raw.iloc[0]) * 100

# Rolling 30-day volatility (annualised)
vol = returns.rolling(30).std() * np.sqrt(252) * 100

# ── EDA summary ───────────────────────────────────────────────────────────────
print("\n=== Annualised Return ===")
total_ret = (raw.iloc[-1] / raw.iloc[0]) ** (1 / 10) - 1
print(total_ret.sort_values(ascending=False).apply(lambda x: f"{x:.1%}"))

print("\n=== Avg Annualised Volatility ===")
print(vol.mean().sort_values(ascending=False).apply(lambda x: f"{x:.1f}%"))

print("\n=== Correlation Matrix ===")
print(returns.corr().round(3))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("S&P 500 Sector EDA — 10-Year Analysis", fontsize=16, fontweight="bold")

palette = sns.color_palette("tab10", len(norm.columns))

# 1. Normalised price performance
for col, color in zip(norm.columns, palette):
    axes[0, 0].plot(norm.index, norm[col], label=col, color=color, linewidth=1.5)
axes[0, 0].set_title("Normalised Performance (Base = 100)")
axes[0, 0].set_ylabel("Index Level")
axes[0, 0].legend(fontsize=8)

# 2. Rolling 30-day volatility
for col, color in zip(vol.columns, palette):
    axes[0, 1].plot(vol.index, vol[col], label=col, color=color, linewidth=1.2, alpha=0.85)
axes[0, 1].set_title("30-Day Rolling Volatility (Annualised %)")
axes[0, 1].set_ylabel("Volatility (%)")
axes[0, 1].legend(fontsize=8)

# 3. Return distribution
returns["S&P 500"].hist(ax=axes[1, 0], bins=80, color="#4fc3f7", edgecolor="none", density=True)
axes[1, 0].axvline(0, color="white", linewidth=1, linestyle="--")
axes[1, 0].set_title("S&P 500 Daily Return Distribution")
axes[1, 0].set_xlabel("Daily Return")
axes[1, 0].set_ylabel("Density")

# 4. Correlation heatmap
corr = returns.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=axes[1, 1], annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, square=True, linewidths=0.5)
axes[1, 1].set_title("Return Correlation Heatmap")

plt.tight_layout()
plt.savefig("stock_market_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to stock_market_analysis.png")
