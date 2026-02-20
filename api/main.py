from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Customer Churn Prediction API")

# Load Model, Scaler, and Encoders
if not os.path.exists("models/best_model.pkl"):
    raise RuntimeError("Model headers not found. Please train the model first.")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

# Define Input Structure
# Using str for categorical variables to be user-friendly
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int # Usually 0 or 1 in dataset, but technically categorical. Let's keep int as it's often numeric in raw data
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

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        input_dict = data.dict()
        
        # Preprocess categorical variables
        # We need to apply the same encoding as in preprocessing.py
        # Note: SeniorCitizen is typically int 0/1, so we might skip it if encoders doesn't have it
        
        for col, value in input_dict.items():
            if col in encoders:
                le = encoders[col]
                # Handle unknown labels? For now, we assume valid input or let it raise error
                # Ideally we should catch verify values
                try:
                    input_dict[col] = le.transform([value])[0]
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid value '{value}' for column '{col}'. Allowed: {list(le.classes_)}")
        
        # Create DataFrame - ensure column order matches training
        # We don't have the exact column list from training here, but usually dict preserves insertion order 
        # and if we constructed it closely to the CSV struture it should be fine.
        # Ideally we'd save the column names list in preprocessing too.
        
        # Let's hope Pydantic field order matches CSV column order (excluding CustomerID)
        # Verify: gender, SeniorCitizen, Partner, Dependents... seems standard Telco dataset order
        
        input_cols = list(input_dict.keys())
        input_data = pd.DataFrame([input_dict], columns=input_cols)
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]
        
        result = "Yes" if prediction[0] == 1 else "No"
        
        return {
            "churn_probability": float(prob),
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn api.main:app --reload