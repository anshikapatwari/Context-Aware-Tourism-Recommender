"""
Step 4: DNN Model Training (PyTorch)
======================================
Builds and trains the Deep Neural Network exactly as described in Paper 28,
implemented using PyTorch instead of TensorFlow.

Paper Reference: Section 3.2
  Architecture:
    - Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, linear)
  Compilation:
    - Loss: MSE | Optimizer: Adam (lr=0.001) | Metrics: MAE
  Training:
    - Epochs: 300 (max) | Batch size: 32 | Shuffle: True
    - EarlyStopping: monitor='val_loss', patience=25, restore_best_weights=True
"""

import os
import copy
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Hyperparameters (matching the paper exactly)
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOP_PATIENCE = 25

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# DEFINE THE DNN MODEL (Paper: Section 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

class RecommenderDNN(nn.Module):
    """
    Deep Neural Network matching Paper 28 architecture:
      Dense(128, ReLU) → Dropout(0.3)
      Dense(64, ReLU)  → Dropout(0.3)
      Dense(1, Linear) — Regression output
    """
    def __init__(self, input_dim, dropout_rate=0.3):
        super(RecommenderDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)       # Linear output for regression
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    EarlyStopping matching Paper 28:
      monitor='val_loss', patience=25, restore_best_weights=True
    """
    def __init__(self, patience=25):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.should_stop = False

    def step(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION — Only runs when this script is called directly,
# NOT when imported by step5/step6/step7
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. LOAD TRAINING AND VALIDATION DATA (Test remains sealed)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4.1: Loading Train & Validation Data")
    print("="*80)

    df_train = pd.read_pickle(os.path.join(DATA_DIR, "train.pkl"))
    df_val   = pd.read_pickle(os.path.join(DATA_DIR, "val.pkl"))

    print(f"  Train: {len(df_train):,} samples")
    print(f"  Val:   {len(df_val):,} samples")
    print(f"  ⚠️  Test set NOT loaded (sealed until Step 6)")
    print(f"  Device: {DEVICE}")


    # ═══════════════════════════════════════════════════════════════════════════
    # 2. PREPARE FEATURES AND TARGET
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4.2: Preparing Feature Matrices")
    print("="*80)

    feature_config = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_config.joblib"))
    feature_cols = feature_config["extended_features"]
    target_col = feature_config["target_column"]
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    print(f"  Using {len(feature_cols)} features: {feature_cols}")
    print(f"  Target: {target_col}")

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train[target_col].values.astype(np.float32)
    X_val   = df_val[feature_cols].values.astype(np.float32)
    y_val   = df_val[target_col].values.astype(np.float32)

    print(f"\n  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  y_val range:   [{y_val.min():.4f}, {y_val.max():.4f}]")

    num_features = X_train.shape[1]


    # ═══════════════════════════════════════════════════════════════════════════
    # 3. CREATE PYTORCH DATALOADERS (shuffle + batch=32)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4.3: Creating PyTorch DataLoaders")
    print("="*80)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True)

    print(f"  Train DataLoader: {len(train_loader)} batches (shuffle=True, batch={BATCH_SIZE})")
    print(f"  Val DataLoader:   {len(val_loader)} batches (shuffle=False, batch={BATCH_SIZE})")


    # ═══════════════════════════════════════════════════════════════════════════
    # 4. BUILD THE DNN MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4.4: Building the Deep Neural Network")
    print("="*80)
    print(f"""
  Architecture (from Paper 28):
  ┌────────────────────────────────────────────┐
  │  Input({num_features})                              │
  │  Linear(128) + ReLU                        │
  │  Dropout({DROPOUT_RATE})                            │
  │  Linear(64) + ReLU                         │
  │  Dropout({DROPOUT_RATE})                            │
  │  Linear(1) — Regression Output             │
  └────────────────────────────────────────────┘
""")

    model = RecommenderDNN(input_dim=num_features, dropout_rate=DROPOUT_RATE).to(DEVICE)

    # Loss and optimizer (Paper: MSE + Adam lr=0.001)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler (additional improvement)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Loss: MSE | Optimizer: Adam (lr={LEARNING_RATE})")
    print(model)


    # ═══════════════════════════════════════════════════════════════════════════
    # 5. TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4.5: Training the DNN")
    print("="*80)
    print(f"  Max epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"  EarlyStopping: patience={EARLY_STOP_PATIENCE}")
    print("-"*80)

    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    history = {
        "loss": [], "val_loss": [],
        "mae": [], "val_mae": []
    }

    for epoch in range(1, EPOCHS + 1):
        # ── Training phase ──
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_mae_sum += torch.sum(torch.abs(y_pred - y_batch)).item()
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_mae = train_mae_sum / train_count

        # ── Validation phase ──
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                batch_size = X_batch.size(0)
                val_loss_sum += loss.item() * batch_size
                val_mae_sum += torch.sum(torch.abs(y_pred - y_batch)).item()
                val_count += batch_size

        val_loss = val_loss_sum / val_count
        val_mae = val_mae_sum / val_count

        # Record history
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        early_stopper.step(val_loss, model)

        # Print progress every 10 epochs or at end
        if epoch % 10 == 0 or epoch == 1 or early_stopper.should_stop:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>4}/{EPOCHS}  |  "
                  f"Train Loss: {train_loss:.6f}  MAE: {train_mae:.6f}  |  "
                  f"Val Loss: {val_loss:.6f}  MAE: {val_mae:.6f}  |  "
                  f"LR: {current_lr:.2e}")

        if early_stopper.should_stop:
            print(f"\n  ⏹ EarlyStopping triggered at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
            early_stopper.restore(model)
            print(f"  ✅ Best weights restored (val_loss={early_stopper.best_loss:.6f})")
            break

    final_epoch = epoch
    best_val_loss = early_stopper.best_loss if early_stopper.best_loss is not None else min(history["val_loss"])
    best_val_mae = min(history["val_mae"])

    print(f"\n  Training completed!")
    print(f"  Epochs run: {final_epoch}")
    print(f"  Best val_loss (MSE): {best_val_loss:.6f}")
    print(f"  Best val_mae:        {best_val_mae:.6f}")


    # ═══════════════════════════════════════════════════════════════════════════
    # 6. SAVE MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4.6: Saving Trained Model")
    print("="*80)

    # Save full model (architecture + weights)
    model_path = os.path.join(MODEL_DIR, "recommender_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": num_features,
        "dropout_rate": DROPOUT_RATE,
        "best_val_loss": best_val_loss,
        "epochs_trained": final_epoch,
        "feature_cols": feature_cols,
    }, model_path)
    print(f"  ✅ Full checkpoint: {model_path}")

    # Save just the weights for easy loading
    weights_path = os.path.join(MODEL_DIR, "recommender_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"  ✅ Weights only: {weights_path}")

    # Export to ONNX for interoperability
    try:
        dummy_input = torch.randn(1, num_features, dtype=torch.float32).to(DEVICE)
        onnx_path = os.path.join(MODEL_DIR, "recommender_model.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True,
            opset_version=11,
            input_names=["features"],
            output_names=["score"],
            dynamic_axes={"features": {0: "batch_size"}, "score": {0: "batch_size"}}
        )
        print(f"  ✅ ONNX format: {onnx_path}")
    except Exception as e:
        print(f"  ⚠️  ONNX export skipped: {e}")


    # ═══════════════════════════════════════════════════════════════════════════
    # 7. SAVE TRAINING HISTORY & PLOTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4.7: Saving Training History & Plots")
    print("="*80)

    joblib.dump(history, os.path.join(ARTIFACTS_DIR, "training_history.joblib"))

    # Plot: Training vs Validation Loss + MAE
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss (MSE)", color="#2196F3", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss (MSE)", color="#F44336", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["mae"], label="Training MAE", color="#4CAF50", linewidth=2)
    plt.plot(history["val_mae"], label="Validation MAE", color="#FF9800", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title("Training vs Validation MAE", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Training curves: {os.path.join(PLOTS_DIR, 'training_curves.png')}")


    # ═══════════════════════════════════════════════════════════════════════════
    # 8. SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("STEP 4 COMPLETE — Model Training Summary (PyTorch)")
    print("="*80)
    print(f"""
  Model Architecture (Paper 28):
    Input({num_features}) → Linear(128) + ReLU → Dropout(0.3)
                 → Linear(64) + ReLU → Dropout(0.3)
                 → Linear(1)

  Training Config:
    Optimizer: Adam (lr={LEARNING_RATE})
    Loss: MSE | Epochs: {final_epoch}/{EPOCHS}
    Batch: {BATCH_SIZE} | EarlyStopping: patience={EARLY_STOP_PATIENCE}
    Device: {DEVICE}

  Results:
    Best Val MSE:  {best_val_loss:.6f}
    Best Val MAE:  {best_val_mae:.6f}

  Saved to: {MODEL_DIR}
  ✅ Ready for Step 5: Model Evaluation (Validation Set)
""")

