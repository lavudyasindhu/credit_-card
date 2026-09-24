"""
main.py
-------
FastAPI backend for the Linear Regression prediction system.

Endpoints:
    GET  /          -> health check
    GET  /data      -> basic dataset information
    POST /train     -> (re)train the model on the latest dataset
    POST /predict   -> predict a target value for a given input feature

Run with:
    uvicorn main:app --reload
(from inside the backend/ folder)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data import load_and_prepare_dataset, FEATURE_COLUMN, TARGET_COLUMN
from model import train_model, predict_value, MODEL_PATH
import os

# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Linear Regression Prediction API",
    description=(
        "A simple ML-powered REST API that trains a Linear Regression model "
        f"to predict '{TARGET_COLUMN}' from '{FEATURE_COLUMN}', and serves "
        "predictions to the HTML/CSS/JS frontend."
    ),
    version="1.0.0",
)

# --- CORS configuration ---
# This allows the frontend (served from a different origin, e.g. opened
# directly as a file, or from a different port) to call this API without
# being blocked by the browser's CORS policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # For a college/demo project, allow all origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS (Pydantic) -- used for automatic validation
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Expected JSON body for POST /predict, e.g. {"feature": 1500}"""
    feature: float = Field(
        ...,
        gt=0,
        description=f"The numerical input value for '{FEATURE_COLUMN}' (must be greater than 0).",
        examples=[1500],
    )


class PredictionResponse(BaseModel):
    prediction: float


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def read_root():
    """Simple health check to confirm the API is running."""
    return {
        "status": "ok",
        "message": "Linear Regression Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/data", tags=["Dataset"])
def get_data_info():
    """
    Returns basic information about the current dataset:
    number of records, feature/target column names, and basic stats.
    Useful for the frontend's 'Dataset Information' section.
    """
    try:
        _, stats = load_and_prepare_dataset()
        return stats
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")


@app.post("/train", tags=["Model"])
def train():
    """
    Loads and preprocesses the latest dataset, trains a fresh Linear Regression
    model, evaluates it, and saves it to disk (trained_model.pkl).
    Call this once at startup, and again any time the dataset changes.
    """
    try:
        result = train_model()
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.post("/predict", response_model=PredictionResponse, tags=["Model"])
def predict(request: PredictionRequest):
    """
    Accepts a JSON body like {"feature": 1500} and returns
    a JSON response like {"prediction": 63.25}.
    """
    try:
        prediction = predict_value(request.feature)
        return PredictionResponse(prediction=round(prediction, 2))
    except FileNotFoundError as e:
        # No trained model yet -> tell the client to call /train first
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ---------------------------------------------------------------------------
# STARTUP: train a model automatically on first launch if one doesn't exist,
# so /predict works immediately without requiring a manual /train call first.
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    if not os.path.exists(MODEL_PATH):
        try:
            train_model()
            print("Startup: no existing model found -> trained a new one automatically.")
        except Exception as e:
            print(f"Startup training skipped/failed: {e}")
