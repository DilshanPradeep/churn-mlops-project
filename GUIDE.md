# Full Project Documentation — Customer Churn MLOps

## Project Files (Click to Open)

| File | Purpose |
|---|---|
| [data_ingestion.py](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/src/data_ingestion.py) | Stage 1 — Loads raw CSV |
| [preprocessing.py](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/src/preprocessing.py) | Stage 2 — Cleans & encodes data |
| [train.py](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/src/train.py) | Stage 3 — Trains 3 ML models |
| [evaluate.py](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/src/evaluate.py) | Stage 4 — Evaluates & picks best model |
| [api/main.py](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/api/main.py) | FastAPI prediction server |
| [airflow_dags/churn_pipeline.py](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/airflow_dags/churn_pipeline.py) | Airflow DAG for automation |
| [dvc.yaml](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/dvc.yaml) | DVC pipeline definition |
| [Dockerfile](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/Dockerfile) | Docker container for API |
| [requirements.txt](file:///d:/SLTC/SEM%207/ML/ML%20Ops/Code/churn-mlops-project/requirements.txt) | Python dependencies |

## Online Links

| Service | URL |
|---|---|
| GitHub Repository | [https://github.com/DilshanPradeep/churn-mlops-project](https://github.com/DilshanPradeep/churn-mlops-project) |
| DAGsHub Project | [https://dagshub.com/DilshanPradeep/churn-mlops-project](https://dagshub.com/DilshanPradeep/churn-mlops-project) |
| MLflow (Local) | http://127.0.0.1:5000 *(run `mlflow ui` first)* |
| API Docs (Local) | http://127.0.0.1:8000/docs *(run `uvicorn api.main:app --reload` first)* |

---

## Step-by-Step Execution

> [!IMPORTANT]
> Every command must be run from inside `d:\SLTC\SEM 7\ML\ML Ops\Code\churn-mlops-project`

### 1. Open Terminal & Navigate

```powershell
cd "d:\SLTC\SEM 7\ML\ML Ops\Code\churn-mlops-project"
```

### 2. Fix Git Path (Every New Terminal)

```powershell
$env:Path += ";C:\Program Files\Git\cmd"
```

### 3. Install All Dependencies (First Time Only)

```bash
pip install -r requirements.txt
pip install dvc-s3
```

### 4. Run the Full ML Pipeline

```bash
dvc repro
```

This runs 4 stages automatically:

```
data_ingestion → data_preprocessing → model_training → model_evaluation
```

✅ Success looks like:
```
Best Model: Logistic_Regression with Accuracy: 0.815...
```

### 5. Push Data to DAGsHub

```bash
dvc push
```

### 6. Push Code to GitHub

```bash
git add .
git commit -m "Complete MLOps pipeline"
git push
```

### 7. View MLflow Experiments

```bash
mlflow ui
```
Open → [http://127.0.0.1:5000](http://127.0.0.1:5000)

Click on **Churn_Prediction_Evaluation** → compare all 3 models.

### 8. Start the Prediction API

```bash
uvicorn api.main:app --reload
```
Open → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Test with this JSON:**
```json
{
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
```

Expected response:
```json
{
  "prediction": "Yes",
  "churn_probability": 0.85,
  "message": "High risk - prepare retention offer!"
}
```

---

## What Each Tool Does

| Tool | Role in This Project |
|---|---|
| **Git** | Tracks code history — every change you make |
| **GitHub** | Cloud backup for your code |
| **DVC** | Tracks data & model files (too large for Git) |
| **DAGsHub** | Cloud backup for data + hosts MLflow tracking server |
| **MLflow** | Records experiment metrics so you can compare models |
| **FastAPI** | Turns your trained model into a live web API |
| **Airflow** | Automates the 4-stage pipeline on a schedule |
| **Docker** | Packages the API so it runs anywhere |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `'git' is not recognized` | Run `$env:Path += ";C:\Program Files\Git\cmd"` |
| `not inside of a DVC repository` | Run `cd "d:\SLTC\SEM 7\ML\ML Ops\Code\churn-mlops-project"` first |
| `No module named dvc_s3` | Run `pip install dvc-s3` |
| API gives 500 error | Run `dvc repro` first to regenerate model files |
| Stage is cached (skipping) | This is **normal and good** — DVC skips unchanged stages |
| `Author identity unknown` | Run `git config --global user.email "your@email.com"` |
