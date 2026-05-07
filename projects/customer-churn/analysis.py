"""
Customer Churn Prediction
Logistic regression + random forest on telecom customer data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

np.random.seed(42)
sns.set_theme(style="darkgrid")
N = 5000

# ── Generate synthetic telecom dataset ──────────────────────────────────────
df = pd.DataFrame({
    "tenure": np.random.randint(1, 72, N),
    "monthly_charges": np.random.uniform(20, 120, N),
    "total_charges": np.random.uniform(100, 8000, N),
    "support_calls": np.random.randint(0, 10, N),
    "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"],
                                       N, p=[0.55, 0.25, 0.20]),
    "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20]),
    "payment_method": np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N),
    "senior_citizen": np.random.choice([0, 1], N, p=[0.84, 0.16]),
})

# Churn probability depends on features
churn_prob = (
    0.05
    + 0.25 * (df["contract_type"] == "Month-to-month")
    - 0.15 * (df["tenure"] / 72)
    + 0.10 * (df["support_calls"] / 10)
    + 0.08 * df["senior_citizen"]
    + 0.05 * (df["internet_service"] == "Fiber optic")
)
df["churn"] = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

print(f"Churn rate: {df['churn'].mean():.1%}")

# ── Feature engineering ───────────────────────────────────────────────────────
df_enc = pd.get_dummies(df, columns=["contract_type", "internet_service",
                                      "payment_method"], drop_first=True)
X = df_enc.drop("churn", axis=1)
y = df_enc["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Models ────────────────────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("\n=== Logistic Regression ===")
print(classification_report(y_test, lr.predict(X_test_s)))
print(f"ROC-AUC: {roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1]):.4f}")

print("\n=== Random Forest ===")
print(classification_report(y_test, rf.predict(X_test)))
print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Customer Churn Prediction", fontsize=16, fontweight="bold")

# 1. Confusion matrix — RF
cm = confusion_matrix(y_test, rf.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
axes[0, 0].set_title("Confusion Matrix — Random Forest")
axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("Actual")

# 2. ROC curves
for model, X_t, label, color in [
    (lr, X_test_s, "Logistic Regression", "#4fc3f7"),
    (rf, X_test, "Random Forest", "#ef9a9a"),
]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_t)[:, 1])
    auc = roc_auc_score(y_test, model.predict_proba(X_t)[:, 1])
    axes[0, 1].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linewidth=2)
axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0, 1].set_title("ROC Curve"); axes[0, 1].set_xlabel("FPR"); axes[0, 1].set_ylabel("TPR")
axes[0, 1].legend(fontsize=9)

# 3. Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
axes[1, 0].barh(feat_imp.index[::-1], feat_imp.values[::-1],
                color=sns.color_palette("viridis", 10))
axes[1, 0].set_title("Top 10 Feature Importances (RF)")
axes[1, 0].set_xlabel("Importance")

# 4. Churn rate by contract type
churn_by_contract = df.groupby("contract_type")["churn"].mean() * 100
axes[1, 1].bar(churn_by_contract.index, churn_by_contract.values,
               color=["#ef9a9a", "#80cbc4", "#ffcc80"])
axes[1, 1].set_title("Churn Rate by Contract Type")
axes[1, 1].set_ylabel("Churn Rate (%)")

plt.tight_layout()
plt.savefig("churn_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to churn_analysis.png")
