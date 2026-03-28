"""
Step 3: Data Splitting — 70% Train / 10% Validation / 20% Test
================================================================
Splits the engineered feature dataset into three subsets.
The test set is saved separately and MUST NOT be loaded until
Step 6 (final evaluation).

Paper Reference: Section 3.2
  - Original paper uses 80/20; user requires 70/10/20
  - First split: 80% train+val / 20% test
  - Second split: 87.5% train / 12.5% val (of the 80%)
  - Result: 70% train, 10% val, 20% test
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ENGINEERED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3.1: Loading Engineered Features")
print("="*80)

df = pd.read_pickle(os.path.join(DATA_DIR, "engineered_features.pkl"))
print(f"  Total samples: {len(df):,}")
print(f"  Columns: {list(df.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT THE DATA — 70/10/20
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3.2: Splitting Data (70% Train / 10% Val / 20% Test)")
print("="*80)

# First split: 80% (train + val) / 20% (test)
df_trainval, df_test = train_test_split(
    df, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
)

# Second split: from the 80%, take 87.5% train / 12.5% val
# 0.125 * 0.80 = 0.10 overall validation
df_train, df_val = train_test_split(
    df_trainval, test_size=0.125, random_state=RANDOM_STATE, shuffle=True
)

print(f"  Train:      {len(df_train):>10,} samples ({100*len(df_train)/len(df):.2f}%)")
print(f"  Validation: {len(df_val):>10,} samples ({100*len(df_val)/len(df):.2f}%)")
print(f"  Test:       {len(df_test):>10,} samples ({100*len(df_test)/len(df):.2f}%)")
print(f"  Total:      {len(df_train)+len(df_val)+len(df_test):>10,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VERIFY NO DATA LEAKAGE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3.3: Verifying Data Isolation (No Leakage)")
print("="*80)

train_idx = set(df_train.index)
val_idx   = set(df_val.index)
test_idx  = set(df_test.index)

assert len(train_idx & val_idx) == 0,   "ERROR: Train-Val overlap detected!"
assert len(train_idx & test_idx) == 0,  "ERROR: Train-Test overlap detected!"
assert len(val_idx & test_idx) == 0,    "ERROR: Val-Test overlap detected!"
assert len(train_idx) + len(val_idx) + len(test_idx) == len(df), "ERROR: Size mismatch!"

print("  ✅ No overlap between train and validation")
print("  ✅ No overlap between train and test")
print("  ✅ No overlap between validation and test")
print("  ✅ Total samples match original dataset")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VERIFY TARGET DISTRIBUTION CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3.4: Target Distribution Across Splits")
print("="*80)

for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"\n  {name} — target_score:")
    print(f"    Mean:   {split_df['target_score'].mean():.4f}")
    print(f"    Std:    {split_df['target_score'].std():.4f}")
    print(f"    Min:    {split_df['target_score'].min():.4f}")
    print(f"    Max:    {split_df['target_score'].max():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE SPLITS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3.5: Saving Split Datasets")
print("="*80)

df_train.to_pickle(os.path.join(DATA_DIR, "train.pkl"))
df_val.to_pickle(os.path.join(DATA_DIR, "val.pkl"))
df_test.to_pickle(os.path.join(DATA_DIR, "test.pkl"))

# Also save as CSV for readability
df_train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
df_val.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
df_test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    fsize = os.path.getsize(os.path.join(DATA_DIR, f"{name}.pkl")) / (1024*1024)
    print(f"  {name}.pkl  →  {len(split_df):>10,} rows  |  {fsize:>6.1f} MB")

print(f"\n  ⚠️  TEST SET IS SEALED — DO NOT LOAD UNTIL step6_final_evaluation.py")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 3 COMPLETE — Data Splitting Summary")
print("="*80)
print(f"""
  ┌───────────────────────────────────────────┐
  │  Split       │  Samples     │  Percentage │
  ├───────────────────────────────────────────┤
  │  Train       │  {len(df_train):>10,}  │     {100*len(df_train)/len(df):.1f}%  │
  │  Validation  │  {len(df_val):>10,}  │     {100*len(df_val)/len(df):.1f}%  │
  │  Test ⛔     │  {len(df_test):>10,}  │     {100*len(df_test)/len(df):.1f}%  │
  └───────────────────────────────────────────┘

  Files saved to: {DATA_DIR}
  ✅ Ready for Step 4: Model Training
""")
