from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer will churn. Use **POST /predict** with customer details.",
    version="1.0.0"
)

# Load Model, Scaler, Encoders, and Feature Columns
if not os.path.exists("models/best_model.pkl"):
    raise RuntimeError("Model not found. Please run 'dvc repro' first.")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

with open("models/feature_columns.json") as f:
    feature_columns = json.load(f)

# ── Input Schema with example ──────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    }

# ── Output Schema ───────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Yes",
                "churn_probability": 0.7821,
                "message": "High risk - prepare retention offer!"
            }
        }
    }

# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. Go to /docs to test."}

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict customer churn",
    description=(
        "Send customer details and receive a churn prediction.\n\n"
        "**prediction**: `Yes` = will churn, `No` = will stay\n\n"
        "**churn_probability**: probability score between 0 and 1\n\n"
        "**message**: risk summary"
    )
)
def predict_churn(data: CustomerData):
    try:
        input_dict = data.dict()

        # Encode categorical columns using saved encoders
        for col, value in input_dict.items():
            if col in encoders:
                le = encoders[col]
                try:
                    input_dict[col] = int(le.transform([value])[0])
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value '{value}' for '{col}'. Allowed: {list(le.classes_)}"
                    )

        # Build DataFrame and reorder columns to match training order
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        return PredictionResponse(
            prediction="Yes" if prediction == 1 else "No",
            churn_probability=round(float(prob), 4),
            message="High risk - prepare retention offer!" if prob > 0.6 else "Low risk customer."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn api.main:app --reload