from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy
import sklearn
import catboost

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_model"
    / "churn_pipeline.pkl"
)


# =============================================================================
# LOAD MODEL
# =============================================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AI-Powered Customer Churn Prediction API",
    description=(
        "Customer churn prediction using "
        "CatBoost with SMOTE."
    ),
    version="1.0.0",
)


# =============================================================================
# INPUT SCHEMA
# =============================================================================

class CustomerData(BaseModel):

    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/")
def root():

    return {
        "message": "AI-Powered Customer Churn Prediction API",
        "status": "running",
        "model": "CatBoost + SMOTE",
        "version": "1.0.0",
    }


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
    }


# =============================================================================
# RUNTIME VERSION DIAGNOSTICS
# =============================================================================

@app.get("/version")
def version():

    return {
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
        "catboost": catboost.__version__,
    }


# =============================================================================
# PREDICTION ENDPOINT
# =============================================================================

@app.post("/predict")
def predict(customer: CustomerData):

    try:

        input_data = pd.DataFrame(
            [{
                "CreditScore": customer.CreditScore,
                "Geography": customer.Geography,
                "Gender": customer.Gender,
                "Age": customer.Age,
                "Tenure": customer.Tenure,
                "Balance": customer.Balance,
                "NumOfProducts": customer.NumOfProducts,
                "HasCrCard": customer.HasCrCard,
                "IsActiveMember": customer.IsActiveMember,
                "EstimatedSalary": customer.EstimatedSalary,
            }]
        )

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        churn_label = (
            "Churn"
            if prediction == 1
            else "No Churn"
        )

        return {
            "prediction": prediction,
            "churn_status": churn_label,
            "churn_probability": round(
                probability,
                4
            ),
            "model": "CatBoost + SMOTE",
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )