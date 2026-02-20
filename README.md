# Customer Churn Prediction MLOps Project

This project implements an end-to-end MLOps pipeline for predicting customer churn.

## Project Structure
- `src/`: Python source code for data ingestion, preprocessing, training, and evaluation.
- `airflow_dags/`: Airflow DAGs for orchestration.
- `api/`: FastAPI application for serving predictions.
- `models/`: Directory where trained models and artifacts are saved.
- `data/`: Directory for raw and processed data.
- `dvc.yaml`: DVC pipeline definition.
- `Dockerfile`: Docker configuration for the API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up DVC/MLflow (Optional but recommended):
   - Configure your DAGsHub credentials if using remote storage/logging.

## Running the Pipeline

### Option 1: Using DVC (Local Dev)
Run the entire pipeline:
```bash
dvc repro
```

### Option 2: Using Airflow
1. Ensure Airflow is installed and initialized.
2. Copy `airflow_dags/churn_pipeline.py` to your Airflow DAGs folder.
3. Trigger the `churn_prediction_pipeline` DAG from the Airflow UI.

## Running the API

1. Start the API server:
   ```bash
   uvicorn api.main:app --reload
   ```

2. Test the endpoint:
   - Go to `http://127.0.0.1:8000/docs` to verify via Swagger UI.
   - Or send a POST request to `/predict`.

## Docker

Build and run the API container:
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
