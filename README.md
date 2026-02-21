# Customer Churn Prediction — MLOps Project

## Project Structure

```
churn-mlops-project/
├── data/
│   ├── raw/                  ← Original CSV goes here
│   └── processed/            ← DVC generates train.csv / test.csv here
├── src/
│   ├── data_ingestion.py     ← Stage 1: Load raw data
│   ├── preprocessing.py      ← Stage 2: Clean, encode, scale data
│   ├── train.py              ← Stage 3: Train 3 ML models
│   └── evaluate.py           ← Stage 4: Test models, pick best
├── airflow_dags/
│   └── churn_pipeline.py     ← Airflow DAG definition
├── api/
│   └── main.py               ← FastAPI prediction server
├── models/                   ← DVC saves .pkl files here
├── dvc.yaml                  ← Pipeline definition
├── Dockerfile                ← For containerized API deployment
└── requirements.txt          ← Python dependencies
```

---

> [!IMPORTANT]
> **ALL commands must be run from inside the project folder.**
> Every time you open a new terminal, first run:
> ```bash
> cd "d:\SLTC\SEM 7\ML\ML Ops\Code\churn-mlops-project"
> ```

---

## STEP 0 — Install Dependencies

```bash
pip install -r requirements.txt
pip install dvc-s3
```

---

## STEP 1 — Fix Git Every New Terminal Session

Git was just installed. Run this in PowerShell every time you open a fresh terminal:

```powershell
$env:Path += ";C:\Program Files\Git\cmd"
```

---

## STEP 2 — Run The Full ML Pipeline

One command runs everything (data loading → preprocessing → training → evaluation):

```bash
dvc repro
```

**What it does, stage by stage:**

| Stage | Script | Input | Output |
|---|---|---|---|
| data_ingestion | `src/data_ingestion.py` | `data/raw/Churn Prediction DataSet.csv` | `data/raw/raw_data.csv` |
| data_preprocessing | `src/preprocessing.py` | `data/raw/raw_data.csv` | `data/processed/train.csv`, `test.csv`, `models/scaler.pkl` |
| model_training | `src/train.py` | `data/processed/train.csv` | `models/Logistic_Regression.pkl`, `Random_Forest.pkl`, `XGBoost.pkl` |
| model_evaluation | `src/evaluate.py` | `data/processed/test.csv` + all `.pkl` files | `models/best_model.pkl`, `metrics.json` |

**Expected final output:**
```
Best Model: Logistic_Regression with Accuracy: 0.815...
```

---

## STEP 3 — View Experiment Results (MLflow)

```bash
mlflow ui
```
Open → http://127.0.0.1:5000

You will see the **Churn_Prediction_Training** and **Churn_Prediction_Evaluation** experiments with all metrics logged (accuracy, precision, recall, F1, ROC-AUC).

---

## STEP 4 — Push Data to DAGsHub (DVC)

```bash
dvc push
```

> Requires your DVC remote to be configured (see setup_guide.md).

---

## STEP 5 — Push Code to GitHub

```bash
git add .
git commit -m "Complete MLOps pipeline"
git push
```

---

## STEP 6 — Run The Prediction API

```bash
uvicorn api.main:app --reload
```

Open → http://127.0.0.1:8000/docs

### Test with this sample JSON (high-risk customer):

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

**Expected response:**
```json
{
  "prediction": "Yes",
  "churn_probability": 0.85,
  "message": "High risk - prepare retention offer!"
}
```

---

## STEP 7 — Build & Run Docker Container (Optional)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `git not recognized` | Run `$env:Path += ";C:\Program Files\Git\cmd"` |
| `dvc not recognized` | Run `pip install dvc` |
| `No module named dvc_s3` | Run `pip install dvc-s3` |
| API 500 error | Run `dvc repro` first to regenerate models |
| `data/raw/raw_data.csv not found` | Run `dvc repro` — this gets auto-generated |
| `models/best_model.pkl not found` | Run `dvc repro` — this gets auto-generated |

---

## MLflow Environment Variables (For DAGsHub Remote)

Run before `dvc repro` to send experiments to DAGsHub:

```powershell
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/DilshanPradeep/churn-mlops-project.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "DilshanPradeep"
$env:MLFLOW_TRACKING_PASSWORD = "YOUR_DAGSHUB_TOKEN"
```
