import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os
import argparse

def train_models(train_data_path, model_dir):
    # Load Data
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"File not found at {train_data_path}")

    train_df = pd.read_csv(train_data_path)
    
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    
    # Define Models to Train [cite: 60]
    models = {
        "Logistic_Regression": LogisticRegression(),
        "Random_Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_score = 0.0 # We can't really score without validation set here if we strictly split. 
    # But usually we do Cross Validation or have a validation set.
    # For this assignment simplicity, let's keep it simple: Just train all, save all? 
    # Or maybe we should allow `train.py` to do CV?
    # The original code did train/test split in preprocessing.
    
    # Let's start MLflow run
    # Set tracking URI - You must set this via env var or here
    # mlflow.set_tracking_uri("...") 
    
    mlflow.set_experiment("Churn_Prediction_Training")

    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Log Parameters
            mlflow.log_param("model_name", name)
            
            # Log Model
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Save locally
            model_path = os.path.join(model_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.csv", help="Path to training data")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    args = parser.parse_args()
    
    train_models(args.data, args.model_dir)