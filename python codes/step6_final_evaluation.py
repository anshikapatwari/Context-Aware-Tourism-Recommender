"""
Step 6: Final Evaluation on Unseen Test Set (PyTorch)
======================================================
FIRST TIME the 20% test set is loaded and used.
Evaluates with both regression metrics AND ranking metrics.

Paper Reference: Section 4.1, 4.2
  Regression: MSE, RMSE, MAE, R², MAPE
  Ranking:    AUC-ROC, Precision@k, Recall@k, NDCG@k
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

K_VALUES = [5, 10, 20]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Import Model Class from step4 ──────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from step4_model_training import RecommenderDNN


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR RANKING METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def precision_at_k(y_true_binary, y_scores, k):
    top_k_idx = np.argsort(y_scores)[-k:][::-1]
    return np.sum(y_true_binary[top_k_idx]) / k

def recall_at_k(y_true_binary, y_scores, k):
    total_relevant = np.sum(y_true_binary)
    if total_relevant == 0:
        return 0.0
    top_k_idx = np.argsort(y_scores)[-k:][::-1]
    return np.sum(y_true_binary[top_k_idx]) / total_relevant

def dcg_at_k(relevance, k):
    relevance = np.array(relevance[:k], dtype=np.float64)
    positions = np.arange(1, len(relevance) + 1)
    return np.sum(relevance / np.log2(positions + 1))

def ndcg_at_k(y_true_binary, y_scores, k):
    top_k_idx = np.argsort(y_scores)[-k:][::-1]
    relevance_at_positions = y_true_binary[top_k_idx]
    dcg = dcg_at_k(relevance_at_positions, k)
    ideal_relevance = np.sort(y_true_binary)[::-1]
    idcg = dcg_at_k(ideal_relevance, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_per_user_ranking_metrics(df, pred_col, true_col, user_col, k_values):
    users = df[user_col].unique()
    metrics = {f"Precision@{k}": [] for k in k_values}
    metrics.update({f"Recall@{k}": [] for k in k_values})
    metrics.update({f"NDCG@{k}": [] for k in k_values})
    threshold = df[true_col].quantile(0.75)
    for user in users:
        user_data = df[df[user_col] == user]
        if len(user_data) < max(k_values):
            continue
        y_true_binary = (user_data[true_col].values >= threshold).astype(int)
        y_scores = user_data[pred_col].values
        if y_true_binary.sum() == 0:
            continue
        for k in k_values:
            if len(user_data) >= k:
                metrics[f"Precision@{k}"].append(precision_at_k(y_true_binary, y_scores, k))
                metrics[f"Recall@{k}"].append(recall_at_k(y_true_binary, y_scores, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(y_true_binary, y_scores, k))
    avg_metrics = {}
    for key, values in metrics.items():
        avg_metrics[key] = np.mean(values) if values else 0.0
    return avg_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL AND UNSEEN TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6.1: Loading Model and UNSEEN Test Set")
print("="*80)
print("  🔓 UNSEALING THE TEST SET FOR THE FIRST TIME")

checkpoint = torch.load(os.path.join(MODEL_DIR, "recommender_model.pth"),
                         map_location=DEVICE, weights_only=False)
model = RecommenderDNN(
    input_dim=checkpoint["input_dim"],
    dropout_rate=checkpoint["dropout_rate"]
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"  ✅ Model loaded")

df_test = pd.read_pickle(os.path.join(DATA_DIR, "test.pkl"))
print(f"  Test samples: {len(df_test):,}")

feature_config = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_config.joblib"))
feature_cols = feature_config["extended_features"]
target_col = feature_config["target_column"]
feature_cols = [c for c in feature_cols if c in df_test.columns]

X_test = df_test[feature_cols].values.astype(np.float32)
y_test = df_test[target_col].values.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GENERATE TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6.2: Generating Test Set Predictions")
print("="*80)

with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_pred = model(X_tensor).cpu().numpy()

df_test = df_test.copy()
df_test["predicted_score"] = y_pred

print(f"  Predictions: {len(y_pred):,}")
print(f"  Pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"  True range: [{y_test.min():.4f}, {y_test.max():.4f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGRESSION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6.3: Regression Metrics on Test Set")
print("="*80)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
epsilon = 1e-8
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  TEST SET REGRESSION METRICS (UNSEEN DATA)                    │
  ├────────────────────────────────────────────────────────────────┤
  │  MSE  (Mean Squared Error):            {mse:>12.6f}           │
  │  RMSE (Root Mean Squared Error):       {rmse:>12.6f}           │
  │  MAE  (Mean Absolute Error):           {mae:>12.6f}           │
  │  R²   (Coefficient of Determination):  {r2:>12.6f}           │
  │  MAPE (Mean Abs Percentage Error):     {mape:>11.4f}%          │
  └────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AUC-ROC (Paper: Section 4.1.6, Figure 4)
# ═══════════════════════════════════════════════════════════════════════════════
print("STEP 6.4: AUC-ROC Analysis")
print("="*80)

relevance_threshold = np.percentile(y_test, 75)
y_binary = (y_test >= relevance_threshold).astype(int)

try:
    auc_score = roc_auc_score(y_binary, y_pred)
    fpr, tpr, _ = roc_curve(y_binary, y_pred)
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  Relevance threshold (75th pctile): {relevance_threshold:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1976D2", linewidth=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linewidth=1.5, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#1976D2")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Test Set (Paper Fig. 4)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ ROC curve: {os.path.join(PLOTS_DIR, 'test_roc_curve.png')}")
except Exception as e:
    auc_score = 0.0
    print(f"  ⚠️  AUC computation error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PRECISION-RECALL CURVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6.5: Precision-Recall Curve")
print("="*80)

try:
    precision_vals, recall_vals, _ = precision_recall_curve(y_binary, y_pred)
    pr_auc = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color="#D32F2F", linewidth=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color="#D32F2F")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ PR curve: {os.path.join(PLOTS_DIR, 'test_pr_curve.png')}")
except Exception as e:
    print(f"  ⚠️  PR curve error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RANKING METRICS: Precision@k, Recall@k, NDCG@k
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6.6: Per-User Ranking Metrics")
print("="*80)

ranking_metrics = compute_per_user_ranking_metrics(
    df_test, "predicted_score", target_col, "user_id", K_VALUES
)

print(f"\n  ┌──────────────────────────────────────────────┐")
print(f"  │  RANKING METRICS (averaged across users)     │")
print(f"  ├──────────────────────────────────────────────┤")
for k in K_VALUES:
    p = ranking_metrics.get(f"Precision@{k}", 0)
    r = ranking_metrics.get(f"Recall@{k}", 0)
    n = ranking_metrics.get(f"NDCG@{k}", 0)
    print(f"  │  k={k:<3}  P@k={p:.4f}  R@k={r:.4f}  NDCG@k={n:.4f} │")
print(f"  └──────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RANKING METRICS BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6.7: Ranking Metrics Visualization")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

prec_vals = [ranking_metrics.get(f"Precision@{k}", 0) for k in K_VALUES]
axes[0].bar([f"k={k}" for k in K_VALUES], prec_vals, color=["#1976D2", "#2196F3", "#64B5F6"])
axes[0].set_title("Precision@k", fontsize=14, fontweight="bold")
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(prec_vals):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

rec_vals = [ranking_metrics.get(f"Recall@{k}", 0) for k in K_VALUES]
axes[1].bar([f"k={k}" for k in K_VALUES], rec_vals, color=["#388E3C", "#4CAF50", "#81C784"])
axes[1].set_title("Recall@k", fontsize=14, fontweight="bold")
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(rec_vals):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

ndcg_vals = [ranking_metrics.get(f"NDCG@{k}", 0) for k in K_VALUES]
axes[2].bar([f"k={k}" for k in K_VALUES], ndcg_vals, color=["#F57C00", "#FF9800", "#FFB74D"])
axes[2].set_title("NDCG@k", fontsize=14, fontweight="bold")
axes[2].set_ylim(0, 1)
axes[2].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(ndcg_vals):
    axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

plt.suptitle("Ranking Metrics on Test Set", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "test_ranking_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Ranking metrics chart: {os.path.join(PLOTS_DIR, 'test_ranking_metrics.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TEST SET RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6.8: Test Set Residual Analysis")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
residuals = y_test - y_pred

axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="#5C6BC0", edgecolors="none")
axes[0].axhline(y=0, color="#D32F2F", linewidth=2, linestyle="--")
axes[0].set_xlabel("Predicted Values", fontsize=12)
axes[0].set_ylabel("Residuals", fontsize=12)
axes[0].set_title("Test Set Residual Plot", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)

max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
axes[1].scatter(y_test, y_pred, alpha=0.3, s=8, color="#00897B", edgecolors="none")
axes[1].plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linewidth=2, linestyle="--")
axes[1].set_xlabel("Actual Values", fontsize=12)
axes[1].set_ylabel("Predicted Values", fontsize=12)
axes[1].set_title(f"Predicted vs Actual (Test R² = {r2:.4f})", fontsize=14, fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "test_residual_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Test residuals: {os.path.join(PLOTS_DIR, 'test_residual_analysis.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SAVE ALL TEST METRICS
# ═══════════════════════════════════════════════════════════════════════════════
all_test_metrics = {
    "regression": {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape},
    "classification": {"AUC_ROC": auc_score},
    "ranking": ranking_metrics,
    "n_samples": len(y_test)
}
joblib.dump(all_test_metrics, os.path.join(ARTIFACTS_DIR, "test_metrics.joblib"))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 6 COMPLETE — Final Test Evaluation Summary")
print("="*80)
print(f"""
  REGRESSION METRICS (Test Set — {len(y_test):,} samples):
    MSE:   {mse:.6f}
    RMSE:  {rmse:.6f}
    MAE:   {mae:.6f}
    R²:    {r2:.6f}
    MAPE:  {mape:.4f}%

  CLASSIFICATION:
    AUC-ROC: {auc_score:.4f}

  RANKING METRICS:""")
for k in K_VALUES:
    p = ranking_metrics.get(f"Precision@{k}", 0)
    r = ranking_metrics.get(f"Recall@{k}", 0)
    n = ranking_metrics.get(f"NDCG@{k}", 0)
    print(f"    k={k}: Precision={p:.4f}, Recall={r:.4f}, NDCG={n:.4f}")
print(f"""
  Plots saved to: {PLOTS_DIR}
  ✅ Ready for Step 7: Hybrid Recommender System
""")
