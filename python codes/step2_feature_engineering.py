"""
Step 2: Feature Engineering & Preprocessing
=============================================
Computes the 3 core features from Paper 28 (distance, rating, cost),
plus additional contextual features from the full Yelp dataset.
Applies imputation (SimpleImputer) and normalization (MinMaxScaler),
then constructs the target variable.

Paper Reference: Section 3.2, 3.4.1, 3.4.2
  - Distance: Haversine formula from user location to business GPS
  - Rating:   Composite popularity score
  - Cost:     RestaurantsPriceRange2 attribute
  - Normalization: MinMaxScaler to [0, 1]
  - Imputation: SimpleImputer with mean strategy
  - Target = w1*(1-norm_distance) + w2*norm_rating + w3*(1-norm_cost)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from math import radians, cos, sin, asin, sqrt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Target weighting (paper: distance most important, then rating, then cost)
W_DISTANCE = 0.50   # proximity is most critical
W_RATING   = 0.35   # popularity second
W_COST     = 0.15   # cost least weight

# Sampling: for very large datasets, sample user-business pairs
MAX_INTERACTIONS = 500_000   # cap to manage memory; set None for all

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD PICKLE DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.1: Loading Pickled DataFrames")
print("="*80)

df_business = pd.read_pickle(os.path.join(DATA_DIR, "business.pkl"))
df_review   = pd.read_pickle(os.path.join(DATA_DIR, "review.pkl"))
df_user     = pd.read_pickle(os.path.join(DATA_DIR, "user.pkl"))
df_checkin  = pd.read_pickle(os.path.join(DATA_DIR, "checkin.pkl"))
df_tip      = pd.read_pickle(os.path.join(DATA_DIR, "tip.pkl"))

print(f"  Business: {len(df_business):,} | Review: {len(df_review):,} | User: {len(df_user):,}")
print(f"  Checkin: {len(df_checkin):,} | Tip: {len(df_tip):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE USER CENTROID (Simulated User Location)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.2: Computing User Location Centroids")
print("="*80)
print("  (Paper uses GPS; we simulate from user's reviewed businesses)")

# Merge review with business to get lat/lon for each review
review_with_loc = df_review[["user_id", "business_id"]].merge(
    df_business[["business_id", "latitude", "longitude"]],
    on="business_id", how="inner"
)

# Compute user centroid (mean lat/lon of all businesses they reviewed)
user_centroids = review_with_loc.groupby("user_id").agg(
    user_lat=("latitude", "mean"),
    user_lon=("longitude", "mean"),
    num_reviews=("business_id", "count")
).reset_index()

print(f"  Users with computed centroids: {len(user_centroids):,}")
print(f"  Avg reviews per user: {user_centroids['num_reviews'].mean():.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD USER-BUSINESS INTERACTION PAIRS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.3: Building User-Business Interaction Pairs")
print("="*80)

# Start from actual reviews (real interactions)
interactions = df_review[["user_id", "business_id", "stars", "useful", "funny", "cool", "date"]].copy()
interactions.rename(columns={"stars": "review_stars"}, inplace=True)

# Sample if too large
if MAX_INTERACTIONS is not None and len(interactions) > MAX_INTERACTIONS:
    print(f"  Sampling {MAX_INTERACTIONS:,} interactions from {len(interactions):,} total")
    interactions = interactions.sample(n=MAX_INTERACTIONS, random_state=42).reset_index(drop=True)
else:
    print(f"  Using all {len(interactions):,} interactions")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MERGE ALL DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.4: Merging All Data Sources")
print("="*80)

# 4a. Merge with business info
df = interactions.merge(
    df_business[["business_id", "name", "latitude", "longitude", "stars",
                  "review_count", "categories", "price_range"]],
    on="business_id", how="left"
)
df.rename(columns={"stars": "business_stars", "review_count": "business_review_count"}, inplace=True)
print(f"  After merging business: {len(df):,} rows")

# 4b. Merge with user centroids
df = df.merge(user_centroids[["user_id", "user_lat", "user_lon"]], on="user_id", how="left")
print(f"  After merging user centroids: {len(df):,} rows")

# 4c. Merge with checkin counts
df = df.merge(df_checkin[["business_id", "checkin_count"]], on="business_id", how="left")
df["checkin_count"] = df["checkin_count"].fillna(0)
print(f"  After merging checkins: {len(df):,} rows")

# 4d. Merge with tip counts (aggregate tips per business)
tip_counts = df_tip.groupby("business_id").agg(
    tip_count=("business_id", "count"),
    avg_tip_compliment=("compliment_count", "mean")
).reset_index()
df = df.merge(tip_counts, on="business_id", how="left")
df["tip_count"] = df["tip_count"].fillna(0)
df["avg_tip_compliment"] = df["avg_tip_compliment"].fillna(0)
print(f"  After merging tips: {len(df):,} rows")

# 4e. Merge with user profile info
user_profile_cols = ["user_id", "review_count", "average_stars", "fans", "useful", "funny", "cool"]
user_profile_cols = [c for c in user_profile_cols if c in df_user.columns]
df_user_subset = df_user[user_profile_cols].copy()
df_user_subset.rename(columns={
    "review_count": "user_review_count",
    "average_stars": "user_avg_stars",
    "useful": "user_useful",
    "funny": "user_funny",
    "cool": "user_cool"
}, inplace=True)
df = df.merge(df_user_subset, on="user_id", how="left")
print(f"  After merging user profile: {len(df):,} rows")
print(f"  Total columns: {len(df.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE FEATURE 1: DISTANCE (Haversine)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.5: Computing Haversine Distance")
print("="*80)
print("  Paper ref: d(u,i) = 2R·arcsin(√(sin²(Δφ/2) + cos(φu)cos(φi)sin²(Δλ/2)))")

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine_vectorized(
    df["user_lat"].values, df["user_lon"].values,
    df["latitude"].values, df["longitude"].values
)

print(f"  Distance stats (km):\n{df['distance_km'].describe().to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPUTE FEATURE 2: COMPOSITE RATING (Popularity)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.6: Computing Composite Rating Score")
print("="*80)
print("  Combines business_stars, review_stars, business_review_count, checkin frequency")

# Normalize components to [0,1] for combination
def safe_minmax(series):
    """Min-max scale a series to [0,1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

norm_biz_stars   = safe_minmax(df["business_stars"].fillna(df["business_stars"].median()))
norm_rev_stars   = safe_minmax(df["review_stars"].fillna(df["review_stars"].median()))
norm_rev_count   = safe_minmax(np.log1p(df["business_review_count"].fillna(0)))
norm_checkin     = safe_minmax(np.log1p(df["checkin_count"]))
norm_tip         = safe_minmax(np.log1p(df["tip_count"]))

# Weighted composite rating
df["composite_rating"] = (
    0.30 * norm_biz_stars +
    0.25 * norm_rev_stars +
    0.20 * norm_rev_count +
    0.15 * norm_checkin +
    0.10 * norm_tip
)

print(f"  Composite rating stats:\n{df['composite_rating'].describe().to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. COMPUTE FEATURE 3: COST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.7: Computing Cost Feature")
print("="*80)
print("  Using RestaurantsPriceRange2 (1-4 scale)")

# price_range already extracted in step1
print(f"  Price range coverage: {df['price_range'].notna().sum():,} / {len(df):,} ({100*df['price_range'].notna().mean():.1f}%)")
print(f"  Distribution:\n{df['price_range'].value_counts(dropna=False).to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ADDITIONAL CONTEXTUAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.8: Computing Additional Context Features")
print("="*80)

# User engagement score
df["user_engagement"] = (
    np.log1p(df["user_review_count"].fillna(0)) +
    np.log1p(df["user_useful"].fillna(0)) +
    np.log1p(df["fans"].fillna(0))
)

# Review helpfulness score
df["review_helpfulness"] = (
    df["useful"].fillna(0) + df["funny"].fillna(0) + df["cool"].fillna(0)
)

# Popularity score (business-level)
df["popularity_score"] = (
    np.log1p(df["business_review_count"].fillna(0)) +
    np.log1p(df["checkin_count"]) +
    np.log1p(df["tip_count"])
)

print("  Computed: user_engagement, review_helpfulness, popularity_score")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. IMPUTATION — SimpleImputer (Paper: mean strategy)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.9: Imputing Missing Values (SimpleImputer, strategy=mean)")
print("="*80)

feature_columns = [
    "distance_km", "composite_rating", "price_range",
    "user_engagement", "review_helpfulness", "popularity_score",
    "user_avg_stars", "review_stars"
]

print(f"  Missing BEFORE imputation:")
for col in feature_columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"    {col}: {missing:,} ({100*missing/len(df):.2f}%)")

imputer = SimpleImputer(strategy="mean")
df[feature_columns] = imputer.fit_transform(df[feature_columns])

# Save imputer
joblib.dump(imputer, os.path.join(ARTIFACTS_DIR, "imputer.joblib"))
print(f"\n  ✅ Imputer saved to: {os.path.join(ARTIFACTS_DIR, 'imputer.joblib')}")

print(f"\n  Missing AFTER imputation:")
remaining = df[feature_columns].isnull().sum().sum()
print(f"    Total NaN remaining: {remaining}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. NORMALIZATION — MinMaxScaler to [0, 1] (Paper: Section 3.4.2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.10: Normalizing Features to [0, 1] (MinMaxScaler)")
print("="*80)

# Columns to normalize
norm_columns = feature_columns.copy()
norm_col_names = ["norm_" + c for c in norm_columns]

scaler = MinMaxScaler(feature_range=(0, 1))
df[norm_col_names] = scaler.fit_transform(df[norm_columns])

# Save scaler
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
print(f"  ✅ Scaler saved to: {os.path.join(ARTIFACTS_DIR, 'scaler.joblib')}")

# Verify all normalized values in [0, 1]
for col in norm_col_names:
    mn, mx = df[col].min(), df[col].max()
    print(f"  {col}: [{mn:.4f}, {mx:.4f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CONSTRUCT TARGET VARIABLE (Paper: Section 3.4.2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.11: Constructing Target Score")
print("="*80)
print(f"  Formula: target = {W_DISTANCE}*(1 - norm_distance) + {W_RATING}*norm_rating + {W_COST}*(1 - norm_cost)")
print("  (Paper: proximity most important, then rating, then cost)")

# Target score — higher means more recommended
df["target_score"] = (
    W_DISTANCE * (1 - df["norm_distance_km"]) +      # closer = better
    W_RATING   * df["norm_composite_rating"] +         # higher rating = better
    W_COST     * (1 - df["norm_price_range"])           # cheaper = better
)

print(f"\n  Target score stats:\n{df['target_score'].describe().to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. DEFINE FEATURE SET FOR MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.12: Defining Final Feature Set for DNN")
print("="*80)

# Core 3 features matching the paper's input vector xi = (d̃, r̃, c̃)
core_features = ["norm_distance_km", "norm_composite_rating", "norm_price_range"]

# Extended features (using everything from the dataset as requested)
extended_features = [
    "norm_distance_km", "norm_composite_rating", "norm_price_range",
    "norm_user_engagement", "norm_review_helpfulness", "norm_popularity_score",
    "norm_user_avg_stars", "norm_review_stars"
]

print(f"  Core features (paper's 3):   {core_features}")
print(f"  Extended features (all {len(extended_features)}): {extended_features}")

# Save feature configuration
feature_config = {
    "core_features": core_features,
    "extended_features": extended_features,
    "target_column": "target_score",
    "weights": {"distance": W_DISTANCE, "rating": W_RATING, "cost": W_COST}
}
joblib.dump(feature_config, os.path.join(ARTIFACTS_DIR, "feature_config.joblib"))


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SAVE FINAL DATASET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2.13: Saving Final Engineered Dataset")
print("="*80)

# Columns to save
save_columns = (
    ["user_id", "business_id", "name", "categories"] +
    feature_columns + norm_col_names +
    ["target_score"]
)
save_columns = [c for c in save_columns if c in df.columns]

df_final = df[save_columns].copy()
df_final.to_pickle(os.path.join(OUTPUT_DIR, "engineered_features.pkl"))
df_final.to_csv(os.path.join(OUTPUT_DIR, "engineered_features.csv"), index=False)

fsize_pkl = os.path.getsize(os.path.join(OUTPUT_DIR, "engineered_features.pkl")) / (1024*1024)
fsize_csv = os.path.getsize(os.path.join(OUTPUT_DIR, "engineered_features.csv")) / (1024*1024)

print(f"  Total samples:  {len(df_final):,}")
print(f"  Total features: {len(df_final.columns)}")
print(f"  Saved: engineered_features.pkl ({fsize_pkl:.1f} MB)")
print(f"  Saved: engineered_features.csv ({fsize_csv:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 2 COMPLETE — Feature Engineering Summary")
print("="*80)
print(f"""
  Paper Feature Mapping:
  ┌────────────────────────────────────────────────────────────┐
  │  Paper Concept         │  Yelp Implementation              │
  ├────────────────────────────────────────────────────────────┤
  │  norm_distance         │  Haversine from user centroid      │
  │  calificación (rating) │  Composite: biz_stars + reviews    │
  │  norm_costo (cost)     │  RestaurantsPriceRange2            │
  └────────────────────────────────────────────────────────────┘

  Target Formula:
    target = {W_DISTANCE}*(1 - norm_dist) + {W_RATING}*norm_rating + {W_COST}*(1 - norm_cost)

  Artifacts saved:
    - {os.path.join(ARTIFACTS_DIR, 'imputer.joblib')}
    - {os.path.join(ARTIFACTS_DIR, 'scaler.joblib')}
    - {os.path.join(ARTIFACTS_DIR, 'feature_config.joblib')}

  ✅ Ready for Step 3: Data Splitting
""")
