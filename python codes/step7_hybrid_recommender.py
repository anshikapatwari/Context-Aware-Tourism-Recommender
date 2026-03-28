"""
Step 7: Hybrid Recommendation System (PyTorch)
================================================
Implements the hybrid recommendation algorithm from Paper 28:
  final_score = 0.6 * model_prediction + 0.4 * user_preference_score

Paper Reference: Section 3.3.3, 3.4.4, Table 1
  - Neural network provides s_NN (60% weight)
  - User preference profile provides s_pref (40% weight)
  - Final ranking sorted by s_final descending
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

# Hybrid weights (Paper: α = 0.6)
ALPHA = 0.6
BETA  = 0.4

NUM_SAMPLE_USERS = 5
TOP_N = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Import Model Class from step4 ──────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from step4_model_training import RecommenderDNN


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA AND MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7.1: Loading Model, Data, and Artifacts")
print("="*80)

checkpoint = torch.load(os.path.join(MODEL_DIR, "recommender_model.pth"),
                         map_location=DEVICE, weights_only=False)
model = RecommenderDNN(
    input_dim=checkpoint["input_dim"],
    dropout_rate=checkpoint["dropout_rate"]
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"  ✅ PyTorch model loaded")

df = pd.read_pickle(os.path.join(DATA_DIR, "engineered_features.pkl"))
print(f"  Full dataset: {len(df):,} interactions")

feature_config = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_config.joblib"))
feature_cols = feature_config["extended_features"]
target_col = feature_config["target_column"]
feature_cols = [c for c in feature_cols if c in df.columns]

df_business = pd.read_pickle(os.path.join(DATA_DIR, "business.pkl"))
df_review = pd.read_pickle(os.path.join(DATA_DIR, "review.pkl"))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD USER CATEGORY PREFERENCE PROFILES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7.2: Building User Category Preference Profiles")
print("="*80)
print("  Paper: p(u, c) ∈ [0, 1] for each user u and category c")

df_business["category_list"] = df_business["categories"].apply(
    lambda x: [c.strip() for c in str(x).split(",")] if pd.notna(x) else []
)

all_categories = df_business["category_list"].explode().dropna()
top_categories = all_categories.value_counts().head(30).index.tolist()
print(f"  Top 30 categories: {top_categories[:10]}...")

review_with_biz = df_review[["user_id", "business_id", "stars"]].merge(
    df_business[["business_id", "category_list"]], on="business_id", how="inner"
)

review_exploded = review_with_biz.explode("category_list")
review_exploded = review_exploded[review_exploded["category_list"].isin(top_categories)]

user_cat_prefs = review_exploded.groupby(["user_id", "category_list"])["stars"].mean().reset_index()
user_cat_prefs.rename(columns={"stars": "avg_category_rating"}, inplace=True)
user_cat_prefs["pref_score"] = user_cat_prefs["avg_category_rating"] / 5.0

pref_dict = user_cat_prefs.set_index(["user_id", "category_list"])["pref_score"].to_dict()

print(f"  User-category preference pairs: {len(user_cat_prefs):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREFERENCE LOOKUP FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_user_preference(user_id, categories_str):
    if pd.isna(categories_str):
        return 0.5
    cats = [c.strip() for c in str(categories_str).split(",")]
    scores = []
    for cat in cats:
        if cat in top_categories:
            score = pref_dict.get((user_id, cat), None)
            if score is not None:
                scores.append(score)
    return np.mean(scores) if scores else 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SELECT SAMPLE USERS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7.3: Selecting Sample Users for Demonstration")
print("="*80)

user_review_counts = df.groupby("user_id").size().reset_index(name="count")
active_users = user_review_counts[user_review_counts["count"] >= 20].sort_values("count", ascending=False)
sample_users = active_users.head(NUM_SAMPLE_USERS)["user_id"].tolist()

print(f"  Selected {len(sample_users)} active users for recommendation demo")
for i, uid in enumerate(sample_users):
    cnt = user_review_counts[user_review_counts["user_id"] == uid]["count"].values[0]
    print(f"    User {i+1}: {uid[:12]}... ({cnt} interactions)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE HYBRID RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7.4: Generating Hybrid Recommendations")
print("="*80)
print(f"  Formula: s_final = {ALPHA} × s_NN + {BETA} × s_pref")

all_recommendations = {}

for user_idx, user_id in enumerate(sample_users):
    print(f"\n  ─── User {user_idx+1}: {user_id[:20]}... ───")

    user_data = df[df["user_id"] == user_id].copy()

    if len(user_data) == 0:
        print(f"    No data for this user, skipping")
        continue

    # Get model predictions (s_NN) using PyTorch
    X_user = user_data[feature_cols].values.astype(np.float32)
    with torch.no_grad():
        X_tensor = torch.tensor(X_user, dtype=torch.float32).to(DEVICE)
        s_nn = model(X_tensor).cpu().numpy()

    # Normalize model scores to [0, 100] scale (matching paper's Table 1)
    s_nn_scaled = (s_nn - s_nn.min()) / (s_nn.max() - s_nn.min() + 1e-8) * 100

    # Compute user preference scores
    s_pref = user_data["categories"].apply(
        lambda cats: get_user_preference(user_id, cats)
    ).values
    s_pref_scaled = s_pref * 100

    # Hybrid score (Paper: Section 3.4.4)
    s_final = ALPHA * s_nn_scaled + BETA * s_pref_scaled

    user_data = user_data.copy()
    user_data["tf_score"] = s_nn_scaled
    user_data["pref_score"] = s_pref_scaled
    user_data["final_score"] = s_final

    user_data = user_data.sort_values("final_score", ascending=False).head(TOP_N)
    user_data["rank"] = range(1, len(user_data) + 1)

    user_data["primary_category"] = user_data["categories"].apply(
        lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown"
    )

    all_recommendations[user_id] = user_data
    print(f"    Generated top-{TOP_N} recommendations")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DISPLAY RECOMMENDATION TABLE (Paper: Table 1)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7.5: Recommendation Tables (matching Paper 28, Table 1)")
print("="*80)

for user_idx, (user_id, rec_df) in enumerate(all_recommendations.items()):
    print(f"\n{'='*100}")
    print(f"  RECOMMENDATIONS FOR USER {user_idx+1}: {user_id[:24]}...")
    print(f"  Formula: final_score = {ALPHA} × DNN_Score + {BETA} × Pref_Score")
    print(f"{'='*100}")
    print(f"  {'Rank':<6} {'Business Name':<35} {'Category':<20} {'DNN Score':<12} {'Pref Score':<12} {'Final Score':<12}")
    print(f"  {'─'*6} {'─'*35} {'─'*20} {'─'*12} {'─'*12} {'─'*12}")

    for _, row in rec_df.iterrows():
        name = str(row.get("name", "Unknown"))[:33]
        cat = str(row.get("primary_category", "N/A"))[:18]
        print(f"  {row['rank']:<6} {name:<35} {cat:<20} {row['tf_score']:<12.2f} {row['pref_score']:<12.2f} {row['final_score']:<12.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7.6: Saving Recommendations")
print("="*80)

all_recs_list = []
for user_id, rec_df in all_recommendations.items():
    save_cols = ["user_id", "business_id", "name", "categories", "primary_category",
                 "rank", "tf_score", "pref_score", "final_score"]
    save_cols = [c for c in save_cols if c in rec_df.columns]
    all_recs_list.append(rec_df[save_cols])

if all_recs_list:
    df_all_recs = pd.concat(all_recs_list, ignore_index=True)
    df_all_recs.to_csv(os.path.join(DATA_DIR, "recommendations.csv"), index=False)
    df_all_recs.to_pickle(os.path.join(DATA_DIR, "recommendations.pkl"))
    print(f"  ✅ Saved: recommendations.csv ({len(df_all_recs)} recommendations)")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATION: HYBRID SCORE COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7.7: Generating Recommendation Visualizations")
print("="*80)

if all_recommendations:
    first_user = list(all_recommendations.keys())[0]
    rec_df = all_recommendations[first_user].head(TOP_N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = [str(n)[:15] for n in rec_df["name"].values]
    tf_contrib = ALPHA * rec_df["tf_score"].values
    pref_contrib = BETA * rec_df["pref_score"].values

    x = np.arange(len(names))
    axes[0].barh(x, tf_contrib, color="#1976D2", label=f"DNN ({int(ALPHA*100)}%)", height=0.6)
    axes[0].barh(x, pref_contrib, left=tf_contrib, color="#F57C00", label=f"Preference ({int(BETA*100)}%)", height=0.6)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_xlabel("Score", fontsize=12)
    axes[0].set_title("Hybrid Score Composition", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(x, rec_df["final_score"].values, color="#388E3C", height=0.6)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].set_xlabel("Final Score", fontsize=12)
    axes[1].set_title("Final Recommendation Ranking", fontsize=14, fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle(f"Hybrid Recommendations — Sample User", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hybrid_recommendations.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Hybrid visualization: {os.path.join(PLOTS_DIR, 'hybrid_recommendations.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 7 COMPLETE — Hybrid Recommender System Summary")
print("="*80)
print(f"""
  Hybrid Algorithm (Paper 28, Section 3.4.4):
    s_final = α × s_NN + (1-α) × s_pref
    α = {ALPHA} (DNN weight) | (1-α) = {BETA} (preference weight)

  User Preference Model:
    - Built from user's review history per business category
    - Top {len(top_categories)} categories tracked
    - p(u,c) ∈ [0, 1] normalized preference score

  Results:
    - {NUM_SAMPLE_USERS} sample users demonstrated
    - Top-{TOP_N} recommendations generated per user
    - Recommendation tables matching Paper 28 Table 1 format

  Files:
    - {os.path.join(DATA_DIR, 'recommendations.csv')}
    - {os.path.join(PLOTS_DIR, 'hybrid_recommendations.png')}

  ══════════════════════════════════════════════
  ✅ PIPELINE COMPLETE — All 7 Steps Finished!
  ══════════════════════════════════════════════
""")
