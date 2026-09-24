"""
model.py
--------
Everything related to the Linear Regression model itself:
    - Train/test split
    - Training the model
    - Evaluating it (R2, MAE, MSE, RMSE)
    - Saving it to disk with joblib
    - Loading it back from disk
    - Making a single prediction
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from data import load_and_prepare_dataset, FEATURE_COLUMN, TARGET_COLUMN

# Where the trained model is saved/loaded from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")


def train_model(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Full training pipeline:
        1. Load + clean dataset
        2. Select feature (X) and target (y)
        3. Train/test split
        4. Train a Linear Regression model
        5. Evaluate on the test set
        6. Save the model to disk with joblib

    Returns a dictionary with dataset info + evaluation metrics,
    which is what POST /train sends back to the frontend.
    """
    # 1. Load + clean
    df, dataset_stats = load_and_prepare_dataset()

    if len(df) < 10:
        raise ValueError(
            "Not enough clean data to train a model (need at least 10 rows)."
        )

    # 2. Feature selection: X is a 2D array (scikit-learn requirement),
    #    y is a 1D array of target values.
    X = df[[FEATURE_COLUMN]].values
    y = df[TARGET_COLUMN].values

    # 3. Train/test split (80% train, 20% test by default)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate on the held-out test set
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "mse": round(float(mse), 2),
        "rmse": round(float(rmse), 2),
    }

    # 6. Save the trained model to disk using joblib
    joblib.dump(model, MODEL_PATH)

    return {
        "message": "Model trained and saved successfully.",
        "dataset_info": dataset_stats,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": metrics,
        "model_coefficients": {
            "slope": round(float(model.coef_[0]), 4),
            "intercept": round(float(model.intercept_), 4),
        },
    }


def load_model() -> LinearRegression:
    """Loads the trained model from disk. Raises an error if it doesn't exist yet."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Call POST /train first to train and save a model."
        )
    return joblib.load(MODEL_PATH)


def predict_value(feature_value: float) -> float:
    """
    Loads the saved model and predicts the target value
    for a single given feature value (e.g. square footage -> price).
    """
    model = load_model()
    # scikit-learn expects a 2D array of shape (n_samples, n_features)
    X_new = np.array([[feature_value]])
    prediction = model.predict(X_new)[0]
    return float(prediction)
