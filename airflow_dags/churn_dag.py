from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add project root to path so airflow can find 'src'
sys.path.append("/path/to/your/churn-mlops-project") 

from src.data_ingestion import load_data
from src.preprocessing import preprocess_data
from src.train import train_models

default_args = {
    'owner': 'student_group',
    'start_date': days_ago(1),
}

with DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-End Churn Prediction Pipeline',
    schedule_interval='@daily',
    catchup=False
) as dag:

    # Task 1: Ingest Data [cite: 89]
    t1 = PythonOperator(
        task_id='ingest_data',
        python_callable=load_data
    )

    # Task 2: Preprocessing [cite: 91]
    t2 = PythonOperator(
        task_id='preprocessing',
        python_callable=preprocess_data
    )

    # Task 3: Training & Evaluation [cite: 92, 93]
    t3 = PythonOperator(
        task_id='train_and_evaluate',
        python_callable=train_models
    )

    # Set Dependencies [cite: 96]
    t1 >> t2 >> t3