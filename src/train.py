import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os
import argparse
import json
import dagshub

def train_models(train_data_path, model_dir, test_data_path=None):
    # Load Training Data
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"File not found at {train_data_path}")

    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]

    # Load Test Data for metric logging (optional)
    X_test, y_test = None, None
    if test_data_path and os.path.exists(test_data_path):
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop("Churn", axis=1)
        y_test = test_df["Churn"]

    # Save feature column names for the API
    os.makedirs(model_dir, exist_ok=True)
    feature_columns = list(X_train.columns)
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)
    print(f"Saved feature_columns.json with {len(feature_columns)} columns")

    # Define Models to Train
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Initialize DagShub MLflow tracking
    dagshub.init(repo_owner="DilshanPradeep", repo_name="churn-mlops-project", mlflow=True)
    mlflow.set_experiment("Churn_Prediction_Training")

    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)

            # Log Parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("train_size", len(X_train))

            # Log Metrics using test data
            if X_test is not None and y_test is not None:
                predicted = model.predict(X_test)
                acc  = accuracy_score(y_test, predicted)
                prec = precision_score(y_test, predicted, zero_division=0)
                rec  = recall_score(y_test, predicted, zero_division=0)
                f1   = f1_score(y_test, predicted, zero_division=0)
                try:
                    roc  = roc_auc_score(y_test, predicted)
                except Exception:
                    roc = 0.0

                mlflow.log_metric("accuracy",  acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall",    rec)
                mlflow.log_metric("f1_score",  f1)
                mlflow.log_metric("roc_auc",   roc)

                print(f"  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}  roc_auc={roc:.4f}")

            # Log Model artifact
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
    parser.add_argument("--data",       default="data/processed/train.csv",  help="Path to training data")
    parser.add_argument("--model-dir",  default="models",                    help="Directory to save models")
    parser.add_argument("--test-data",  default="data/processed/test.csv",   help="Path to test data for metric logging")
    args = parser.parse_args()

    train_models(args.data, args.model_dir, args.test_data)