import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import argparse

def preprocess_data(input_path, output_dir):
    # 1. Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at {input_path}")

    df = pd.read_csv(input_path)
    
    # 2. Handle Missing Values & TotalCharges [cite: 54]
    # Blank spaces numbers walata convert karanawa, bari ewa NaN wenawa
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop CustomerID (model ekata wadak na)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    
    # 3. Encode Categorical Variables [cite: 55]
    # Binary columns (Yes/No) label encoding karanawa
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # API eka use karaddi ona wena nisa encoders save karagamu (optional but good practice)
    encoders = {}
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # 4. Feature Scaling [cite: 56]
    target_col = 'Churn'
    if target_col not in df.columns:
         # Handle case where Churn might be named differently or already encoded
         # For now assume it is 'Churn'
         pass

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for the API later
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Save encoders for API
    joblib.dump(encoders, "models/encoders.pkl")
    
    # Convert back to dataframe to save
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)
    X_processed[target_col] = y.values
    
    # 5. Train-Test Split [cite: 57]
    train, test = train_test_split(X_processed, test_size=0.2, random_state=42)
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Preprocessing Completed. Train/Test files saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/raw_data.csv", help="Path to raw data")
    parser.add_argument("--output", default="data/processed", help="Directory to save processed data")
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)