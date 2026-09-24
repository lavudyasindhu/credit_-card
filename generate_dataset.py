"""
generate_dataset.py
--------------------
Generates dataset/data.csv : a House Size (sq. ft.) vs House Price dataset.

WHY THIS APPROACH:
Live real-estate price APIs (Zillow, Realtor.com, MagicBricks, etc.) all require
paid API keys / registration and are not reachable from a locked-down sandbox
network. To keep this project 100% runnable out-of-the-box, WITHOUT any API key
or internet dependency, this script generates a realistic dataset using real
market statistics as its parameters:

    - Average price per square foot in a typical mid-size city market
      (used here: roughly $120-$180 per sq. ft., a realistic US suburban range)
    - A realistic base/fixed cost component (land, permits, foundation, etc.)
    - Realistic random noise to simulate real-world variation
      (location, finish quality, negotiation, market timing, etc.)

This is the same technique real data-science courses use when a live paid
API is not available: model the data generation process on real, published
market statistics, so the relationship you learn (price per sq. ft.) is the
same relationship you would find in a real listings dataset.

HOW TO USE A REAL LIVE DATASET INSTEAD (optional, if you have internet/API access):
    1. Get a CSV of real housing data (e.g. Kaggle "USA Housing", a MLS export,
       or a real-estate API response) with a "SquareFeet" and "Price" column.
    2. Replace dataset/data.csv with that file (keep the same column names,
       or update backend/data.py's FEATURE_COLUMN / TARGET_COLUMN constants).
    3. Call POST /train again -- the backend re-loads dataset/data.csv fresh
       every time you train, so no code changes are needed.

Run this file any time to regenerate a fresh dataset:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

# Reproducible randomness so results are consistent across runs/demos
np.random.seed(42)

N_RECORDS = 500

# Realistic house sizes (square feet) - typical residential range
square_feet = np.random.uniform(500, 4500, N_RECORDS)

# Realistic pricing model based on published US market averages:
#   price = fixed_base_cost + (price_per_sqft * size) + noise
BASE_COST = 15000          # land, permits, foundation etc. (fixed cost)
PRICE_PER_SQFT = 148.0     # realistic average $/sq.ft.
NOISE_STD = 18000          # variation from location/finish/negotiation

noise = np.random.normal(0, NOISE_STD, N_RECORDS)
price = BASE_COST + (PRICE_PER_SQFT * square_feet) + noise

# Prices can't be negative or unrealistically low - clip to a sensible floor
price = np.clip(price, 20000, None)

df = pd.DataFrame({
    "SquareFeet": np.round(square_feet, 1),
    "Price": np.round(price, 2)
})

# --- Intentionally inject a few realistic data-quality issues ---
# so the ML pipeline's cleaning step (missing values, duplicates) has real work to do.

# 1) A handful of missing values (simulates incomplete listings)
missing_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[missing_idx, "Price"] = np.nan

# 2) A handful of exact duplicate rows (simulates duplicate listings)
duplicate_rows = df.sample(5, random_state=1)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Shuffle rows so duplicates/missing values aren't clustered at the end
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

output_path = os.path.join(os.path.dirname(__file__), "data.csv")
df.to_csv(output_path, index=False)

print(f"Dataset generated: {output_path}")
print(f"Rows: {len(df)}")
print(df.head())
