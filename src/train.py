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
import json
import yaml


def load_params(params_file="params.yaml"):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params["model"]


def get_model(model_params):
    name = model_params["name"]
    if name == "Logistic_Regression":
        return LogisticRegression(max_iter=model_params.get("max_iter", 1000))
    elif name == "Random_Forest":
        return RandomForestClassifier(n_estimators=model_params.get("n_estimators", 100))
    elif name == "XGBoost":
        return XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            max_depth=model_params.get("max_depth", 6),
            learning_rate=model_params.get("learning_rate", 0.1),
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_models(train_data_path, model_dir, params_file="params.yaml"):
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"File not found at {train_data_path}")

    model_params = load_params(params_file)
    model_name = model_params["name"]

    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]

    # Save feature column names for the API
    os.makedirs(model_dir, exist_ok=True)
    feature_columns = list(X_train.columns)
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)
    print(f"Saved feature_columns.json with {len(feature_columns)} columns")

    mlflow.set_experiment("Churn_Prediction_Training")

    model = get_model(model_params)

    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Log all params
        mlflow.log_param("model_name", model_name)
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        # Log Model to MLflow
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Save locally for DVC
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {model_name} to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.csv", help="Path to training data")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    train_models(args.data, args.model_dir, args.params)