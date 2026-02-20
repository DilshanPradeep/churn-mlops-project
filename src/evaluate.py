import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import argparse
import json

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    try:
        roc_auc = roc_auc_score(actual, pred)
    except:
        roc_auc = 0.0
    return accuracy, precision, recall, f1, roc_auc

def evaluate_models(model_dir, test_data_path, output_metrics_file):
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"File not found at {test_data_path}")
        
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]
    
    # Find all models in the directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and f != "scaler.pkl"]
    
    best_accuracy = 0.0
    best_model_name = ""
    
    mlflow.set_experiment("Churn_Prediction_Evaluation")
    
    metrics_summary = {}
    
    for model_file in model_files:
        name = model_file.replace(".pkl", "")
        model_path = os.path.join(model_dir, model_file)
        
        print(f"Evaluating {name}...")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load {model_file}: {e}")
            continue
            
        predicted = model.predict(X_test)
        
        (acc, prec, rec, f1, roc) = eval_metrics(y_test, predicted)
        
        metrics_summary[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        }
        
        with mlflow.start_run(run_name=f"Eval_{name}"):
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            # Save as best_model.pkl for API
            joblib.dump(model, os.path.join(model_dir, "best_model.pkl"))
            
    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")
    
    # Save metrics to file (for DVC)
    with open(output_metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models", help="Directory containing trained models")
    parser.add_argument("--test-data", default="data/processed/test.csv", help="Path to test data")
    parser.add_argument("--metrics-file", default="metrics.json", help="File to save evaluation metrics")
    args = parser.parse_args()
    
    evaluate_models(args.model_dir, args.test_data, args.metrics_file)
