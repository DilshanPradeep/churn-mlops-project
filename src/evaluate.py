import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import argparse
import json
import yaml


def load_params(params_file="params.yaml"):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params["model"]


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    try:
        roc_auc = roc_auc_score(actual, pred)
    except Exception:
        roc_auc = 0.0
    return accuracy, precision, recall, f1, roc_auc


def evaluate_models(model_dir, test_data_path, metrics_dir, params_file="params.yaml"):
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"File not found at {test_data_path}")

    model_params = load_params(params_file)
    model_name = model_params["name"]

    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    print(f"Evaluating {model_name}...")
    model = joblib.load(model_path)
    predicted = model.predict(X_test)

    acc, prec, rec, f1, roc = eval_metrics(y_test, predicted)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
    }

    mlflow.set_experiment("Churn_Prediction_Evaluation")
    with mlflow.start_run(run_name=f"Eval_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

    # Write per-model metrics file (DVC tracks this)
    os.makedirs(metrics_dir, exist_ok=True)
    output_metrics_file = os.path.join(metrics_dir, f"{model_name}.json")
    with open(output_metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Saved metrics to {output_metrics_file}")

    # Also save as best_model.pkl if it's the best so far (based on accuracy)
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    best_accuracy = 0.0
    best_metrics_glob = [
        f for f in os.listdir(metrics_dir) if f.endswith(".json")
    ]
    for mf in best_metrics_glob:
        with open(os.path.join(metrics_dir, mf)) as f:
            m = json.load(f)
            if m.get("accuracy", 0) > best_accuracy:
                best_accuracy = m["accuracy"]

    if acc >= best_accuracy:
        joblib.dump(model, best_model_path)
        print(f"Updated best_model.pkl with {model_name} (accuracy={acc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models", help="Directory containing trained models")
    parser.add_argument("--test-data", default="data/processed/test.csv", help="Path to test data")
    parser.add_argument("--metrics-dir", default="metrics", help="Directory to save per-model metric files")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    evaluate_models(args.model_dir, args.test_data, args.metrics_dir, args.params)
