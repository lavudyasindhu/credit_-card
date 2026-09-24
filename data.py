"""
data.py
-------
Handles everything related to the dataset:
    - Loading the CSV from disk
    - Cleaning it (missing values, duplicates, type checking)
    - Basic exploratory statistics
    - Splitting into feature (X) and target (y)

Keeping all "dataset logic" in one file makes it easy to swap in a different
dataset later -- you only need to change FEATURE_COLUMN / TARGET_COLUMN and
DATASET_PATH (or the CSV itself), nothing else in the project needs to change.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Path to the dataset CSV (relative to this file, so it works from any folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "data.csv")

# Column names used for training.
# FEATURE_COLUMN = the single numerical input (X)
# TARGET_COLUMN  = the numerical value we want to predict (y)
FEATURE_COLUMN = "SquareFeet"
TARGET_COLUMN = "Price"


def load_raw_dataset() -> pd.DataFrame:
    """Load the dataset CSV exactly as it is on disk (no cleaning yet)."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            f"Run 'python dataset/generate_dataset.py' first."
        )
    return pd.read_csv(DATASET_PATH)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataset:
        1. Ensures correct data types (numeric columns are actually numeric)
        2. Removes rows with missing values in the columns we care about
        3. Removes duplicate rows
    Returns a clean copy, ready for training.
    """
    df = df.copy()

    # --- 1. Data type checking / correction ---
    # Force the feature and target columns to numeric.
    # Any value that can't be converted becomes NaN, which we then drop.
    df[FEATURE_COLUMN] = pd.to_numeric(df[FEATURE_COLUMN], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    # --- 2. Missing-value handling ---
    # For a simple linear regression demo, the safest approach is to drop rows
    # that are missing the feature or target, rather than guessing/imputing
    # values for a small dataset (imputing could distort the linear relationship).
    df = df.dropna(subset=[FEATURE_COLUMN, TARGET_COLUMN])

    # --- 3. Duplicate removal ---
    df = df.drop_duplicates()

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    return df


def get_basic_stats(df: pd.DataFrame) -> dict:
    """
    Returns basic exploratory statistics about the (cleaned) dataset.
    Used by the GET /data endpoint so the frontend can display dataset info.
    """
    return {
        "total_records": int(len(df)),
        "feature_column": FEATURE_COLUMN,
        "target_column": TARGET_COLUMN,
        "feature_stats": {
            "min": float(df[FEATURE_COLUMN].min()),
            "max": float(df[FEATURE_COLUMN].max()),
            "mean": round(float(df[FEATURE_COLUMN].mean()), 2),
        },
        "target_stats": {
            "min": float(df[TARGET_COLUMN].min()),
            "max": float(df[TARGET_COLUMN].max()),
            "mean": round(float(df[TARGET_COLUMN].mean()), 2),
        },
    }


def load_and_prepare_dataset():
    """
    Convenience function used by model.py:
    loads the raw CSV, cleans it, and returns (clean_df, stats_dict).
    """
    raw_df = load_raw_dataset()
    clean_df = clean_dataset(raw_df)
    stats = get_basic_stats(clean_df)
    return clean_df, stats
