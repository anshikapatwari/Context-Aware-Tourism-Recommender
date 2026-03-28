"""
Step 5: Model Evaluation on Validation Set (PyTorch)
=====================================================
Evaluates the trained PyTorch DNN on the 10% validation dataset.
The test set remains completely unseen.

Paper Reference: Section 4.1, 4.2
  Regression Metrics: RMSE, MAE, MSE, R², MAPE
  Visualizations: Loss curves, residual plot, predicted vs actual
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
    r2_score
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Import Model Class from step4 ──────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from step4_model_training import RecommenderDNN


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL AND VALIDATION DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 5.1: Loading PyTorch Model and Validation Data")
print("="*80)

# Load checkpoint
checkpoint = torch.load(os.path.join(MODEL_DIR, "recommender_model.pth"),
                         map_location=DEVICE, weights_only=False)
model = RecommenderDNN(
    input_dim=checkpoint["input_dim"],
    dropout_rate=checkpoint["dropout_rate"]
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"  ✅ Model loaded (trained for {checkpoint['epochs_trained']} epochs)")

df_val = pd.read_pickle(os.path.join(DATA_DIR, "val.pkl"))
print(f"  Validation samples: {len(df_val):,}")
print(f"  ⚠️  Test set remains SEALED")

# Load feature config
feature_config = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_config.joblib"))
feature_cols = feature_config["extended_features"]
target_col = feature_config["target_column"]
feature_cols = [c for c in feature_cols if c in df_val.columns]

X_val = df_val[feature_cols].values.astype(np.float32)
y_val = df_val[target_col].values.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GENERATE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 5.2: Generating Predictions on Validation Set")
print("="*80)

with torch.no_grad():
    X_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_pred = model(X_tensor).cpu().numpy()

print(f"  Predictions generated: {len(y_pred):,}")
print(f"  Pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"  True range: [{y_val.min():.4f}, {y_val.max():.4f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE REGRESSION METRICS (Paper: Section 4.1.1-4.1.5)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 5.3: Computing Regression Metrics")
print("="*80)

mse  = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_val, y_pred)
r2   = r2_score(y_val, y_pred)

epsilon = 1e-8
mape = np.mean(np.abs((y_val - y_pred) / (y_val + epsilon))) * 100

print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  VALIDATION SET REGRESSION METRICS                            │
  ├────────────────────────────────────────────────────────────────┤
  │  MSE  (Mean Squared Error):            {mse:>12.6f}           │
  │  RMSE (Root Mean Squared Error):       {rmse:>12.6f}           │
  │  MAE  (Mean Absolute Error):           {mae:>12.6f}           │
  │  R²   (Coefficient of Determination):  {r2:>12.6f}           │
  │  MAPE (Mean Abs Percentage Error):     {mape:>11.4f}%          │
  └────────────────────────────────────────────────────────────────┘

  Paper's reported values (on their ontology data):
    RMSE = 0.1955, MAE = 0.0508, MSE = 0.0039, R² = 0.9959
  Note: Different dataset, so exact match is not expected.
""")

# Save metrics
val_metrics = {
    "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape,
    "dataset": "validation", "n_samples": len(y_val)
}
joblib.dump(val_metrics, os.path.join(ARTIFACTS_DIR, "val_metrics.joblib"))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION: TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════════
print("STEP 5.4: Generating Visualizations")
print("="*80)

history = joblib.load(os.path.join(ARTIFACTS_DIR, "training_history.joblib"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["loss"], label="Train Loss", color="#1976D2", linewidth=2)
axes[0].plot(history["val_loss"], label="Val Loss", color="#D32F2F", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss (MSE)", fontsize=12)
axes[0].set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["mae"], label="Train MAE", color="#388E3C", linewidth=2)
axes[1].plot(history["val_mae"], label="Val MAE", color="#F57C00", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("MAE", fontsize=12)
axes[1].set_title("Training vs Validation MAE", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "val_loss_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Loss curves: {os.path.join(PLOTS_DIR, 'val_loss_curves.png')}")


# Residual Analysis (Paper: Figure 3)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
residuals = y_val - y_pred

axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="#5C6BC0", edgecolors="none")
axes[0].axhline(y=0, color="#D32F2F", linewidth=2, linestyle="--")
axes[0].set_xlabel("Predicted Values", fontsize=12)
axes[0].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
axes[0].set_title("Residual Plot (Paper Fig. 3)", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)

max_val = max(y_val.max(), y_pred.max())
min_val = min(y_val.min(), y_pred.min())
axes[1].scatter(y_val, y_pred, alpha=0.3, s=8, color="#00897B", edgecolors="none")
axes[1].plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linewidth=2, linestyle="--", label="Perfect Prediction")
axes[1].set_xlabel("Actual Values", fontsize=12)
axes[1].set_ylabel("Predicted Values", fontsize=12)
axes[1].set_title(f"Predicted vs Actual (R² = {r2:.4f})", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "val_residual_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Residual analysis: {os.path.join(PLOTS_DIR, 'val_residual_analysis.png')}")

# Residual Distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuals, bins=60, color="#7E57C2", edgecolor="white", alpha=0.85)
ax.axvline(x=0, color="#D32F2F", linewidth=2, linestyle="--")
ax.set_xlabel("Residual", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Distribution of Residuals", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "val_residual_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Residual distribution: {os.path.join(PLOTS_DIR, 'val_residual_distribution.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 5 COMPLETE — Validation Evaluation Summary")
print("="*80)
print(f"""
  Metrics (Validation Set — {len(y_val):,} samples):
    MSE:   {mse:.6f}
    RMSE:  {rmse:.6f}
    MAE:   {mae:.6f}
    R²:    {r2:.6f}
    MAPE:  {mape:.4f}%

  Plots saved to: {PLOTS_DIR}
    - val_loss_curves.png
    - val_residual_analysis.png
    - val_residual_distribution.png

  ⚠️  Test set remains SEALED for Step 6
  ✅ Ready for Step 6: Final Evaluation (Unseen Test Set)
""")
