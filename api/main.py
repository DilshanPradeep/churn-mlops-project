from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json

app = FastAPI(title="Customer Churn Prediction API")

# Load Model, Scaler, Encoders, and Feature Columns
if not os.path.exists("models/best_model.pkl"):
    raise RuntimeError("Model not found. Please run 'dvc repro' first.")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

with open("models/feature_columns.json") as f:
    feature_columns = json.load(f)

# Define Input Structure
class CustomerData(BaseModel):
    gender: str          # "Male" or "Female"
    SeniorCitizen: int   # 0 or 1
    Partner: str         # "Yes" or "No"
    Dependents: str      # "Yes" or "No"
    tenure: int          # Number of months
    PhoneService: str    # "Yes" or "No"
    MultipleLines: str   # "Yes", "No", "No phone service"
    InternetService: str # "DSL", "Fiber optic", "No"
    OnlineSecurity: str  # "Yes", "No", "No internet service"
    OnlineBackup: str    # "Yes", "No", "No internet service"
    DeviceProtection: str# "Yes", "No", "No internet service"
    TechSupport: str     # "Yes", "No", "No internet service"
    StreamingTV: str     # "Yes", "No", "No internet service"
    StreamingMovies: str # "Yes", "No", "No internet service"
    Contract: str        # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str# "Yes" or "No"
    PaymentMethod: str   # "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. Go to /docs to test."}

@app.post("/predict")
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
        input_df = input_df[feature_columns]  # Ensure correct column order
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(prob), 4),
            "message": "High risk - prepare retention offer!" if prob > 0.6 else "Low risk customer."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn api.main:app --reload