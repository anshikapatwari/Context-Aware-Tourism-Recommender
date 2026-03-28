"""
Step 1: Data Loading & Exploration
===================================
Loads all 5 Yelp Academic Dataset JSON files (NDJSON format),
explores shapes, missing values, and key columns, then saves
processed DataFrames as pickle files for faster downstream use.

Paper Reference: Section 3.2 — Data Generation and Preprocessing
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helper: Load NDJSON line-by-line (handles multi-GB files) ───────────────
def load_ndjson(filepath, desc="Loading"):
    """Load a newline-delimited JSON file into a list of dicts."""
    records = []
    total_lines = sum(1 for _ in open(filepath, "r", encoding="utf-8"))
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc=desc):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BUSINESS DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1.1: Loading Business Data")
print("="*80)

business_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")
business_records = load_ndjson(business_path, desc="Business")
df_business = pd.DataFrame(business_records)

# Extract key columns
business_cols = [
    "business_id", "name", "city", "state", "latitude", "longitude",
    "stars", "review_count", "is_open", "categories", "attributes"
]
# Keep only columns that exist
business_cols = [c for c in business_cols if c in df_business.columns]
df_business = df_business[business_cols].copy()

print(f"\n  Business records: {len(df_business):,}")
print(f"  Columns: {list(df_business.columns)}")
print(f"  Missing values:\n{df_business.isnull().sum().to_string()}")
print(f"\n  Stars distribution:\n{df_business['stars'].describe().to_string()}")

# Extract RestaurantsPriceRange2 from attributes dict
def extract_price_range(attr):
    """Extract price range from the attributes field."""
    if attr is None or not isinstance(attr, dict):
        return np.nan
    price = attr.get("RestaurantsPriceRange2")
    if price is None:
        return np.nan
    try:
        # Sometimes it's stored as a string like "'2'" or "2"
        return int(str(price).strip("'\""))
    except (ValueError, TypeError):
        return np.nan

df_business["price_range"] = df_business["attributes"].apply(extract_price_range)
print(f"\n  Price range distribution:\n{df_business['price_range'].value_counts(dropna=False).to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD REVIEW DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1.2: Loading Review Data (this may take several minutes)")
print("="*80)

review_path = os.path.join(DATA_DIR, "yelp_academic_dataset_review.json")
review_records = load_ndjson(review_path, desc="Reviews")
df_review = pd.DataFrame(review_records)

review_cols = ["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "date", "text"]
review_cols = [c for c in review_cols if c in df_review.columns]
df_review = df_review[review_cols].copy()

# Convert date to datetime
if "date" in df_review.columns:
    df_review["date"] = pd.to_datetime(df_review["date"], errors="coerce")

print(f"\n  Review records: {len(df_review):,}")
print(f"  Columns: {list(df_review.columns)}")
print(f"  Missing values:\n{df_review.isnull().sum().to_string()}")
print(f"\n  Review stars distribution:\n{df_review['stars'].value_counts().sort_index().to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOAD USER DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1.3: Loading User Data (this may take several minutes)")
print("="*80)

user_path = os.path.join(DATA_DIR, "yelp_academic_dataset_user.json")
user_records = load_ndjson(user_path, desc="Users")
df_user = pd.DataFrame(user_records)

user_cols = [
    "user_id", "name", "review_count", "yelping_since", "useful", "funny", "cool",
    "fans", "average_stars", "compliment_hot", "compliment_more", "compliment_profile",
    "compliment_cute", "compliment_list", "compliment_note", "compliment_plain",
    "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"
]
user_cols = [c for c in user_cols if c in df_user.columns]
df_user = df_user[user_cols].copy()

# Convert yelping_since to datetime
if "yelping_since" in df_user.columns:
    df_user["yelping_since"] = pd.to_datetime(df_user["yelping_since"], errors="coerce")

print(f"\n  User records: {len(df_user):,}")
print(f"  Columns: {list(df_user.columns)}")
print(f"  Missing values:\n{df_user.isnull().sum().to_string()}")
print(f"\n  Average stars distribution:\n{df_user['average_stars'].describe().to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOAD CHECKIN DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1.4: Loading Checkin Data")
print("="*80)

checkin_path = os.path.join(DATA_DIR, "yelp_academic_dataset_checkin.json")
checkin_records = load_ndjson(checkin_path, desc="Checkins")
df_checkin = pd.DataFrame(checkin_records)

# Count total checkins per business
def count_checkins(date_str):
    """Count the number of checkin timestamps from comma-separated string."""
    if pd.isna(date_str) or not date_str:
        return 0
    return len(str(date_str).split(","))

df_checkin["checkin_count"] = df_checkin["date"].apply(count_checkins)
df_checkin = df_checkin[["business_id", "checkin_count"]].copy()

print(f"\n  Checkin records: {len(df_checkin):,}")
print(f"  Checkin count stats:\n{df_checkin['checkin_count'].describe().to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LOAD TIP DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1.5: Loading Tip Data")
print("="*80)

tip_path = os.path.join(DATA_DIR, "yelp_academic_dataset_tip.json")
tip_records = load_ndjson(tip_path, desc="Tips")
df_tip = pd.DataFrame(tip_records)

tip_cols = ["user_id", "business_id", "text", "date", "compliment_count"]
tip_cols = [c for c in tip_cols if c in df_tip.columns]
df_tip = df_tip[tip_cols].copy()

if "date" in df_tip.columns:
    df_tip["date"] = pd.to_datetime(df_tip["date"], errors="coerce")

print(f"\n  Tip records: {len(df_tip):,}")
print(f"  Columns: {list(df_tip.columns)}")
print(f"  Missing values:\n{df_tip.isnull().sum().to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE ALL DATAFRAMES AS PICKLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1.6: Saving Processed DataFrames")
print("="*80)

df_business.to_pickle(os.path.join(OUTPUT_DIR, "business.pkl"))
df_review.to_pickle(os.path.join(OUTPUT_DIR, "review.pkl"))
df_user.to_pickle(os.path.join(OUTPUT_DIR, "user.pkl"))
df_checkin.to_pickle(os.path.join(OUTPUT_DIR, "checkin.pkl"))
df_tip.to_pickle(os.path.join(OUTPUT_DIR, "tip.pkl"))

print(f"\n  Saved 5 pickle files to: {OUTPUT_DIR}")
for name, df in [("business", df_business), ("review", df_review),
                  ("user", df_user), ("checkin", df_checkin), ("tip", df_tip)]:
    fpath = os.path.join(OUTPUT_DIR, f"{name}.pkl")
    fsize = os.path.getsize(fpath) / (1024*1024)
    print(f"    {name}.pkl  →  {len(df):>10,} rows  |  {fsize:>8.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STEP 1 COMPLETE — Data Loading Summary")
print("="*80)
print(f"""
  ┌──────────────────────────────────────────────┐
  │  Dataset          │  Rows        │  Columns  │
  ├──────────────────────────────────────────────┤
  │  Business         │  {len(df_business):>10,}  │  {len(df_business.columns):>7}  │
  │  Review           │  {len(df_review):>10,}  │  {len(df_review.columns):>7}  │
  │  User             │  {len(df_user):>10,}  │  {len(df_user.columns):>7}  │
  │  Checkin          │  {len(df_checkin):>10,}  │  {len(df_checkin.columns):>7}  │
  │  Tip              │  {len(df_tip):>10,}  │  {len(df_tip.columns):>7}  │
  └──────────────────────────────────────────────┘

  All pickle files saved to: {OUTPUT_DIR}
  ✅ Ready for Step 2: Feature Engineering
""")
