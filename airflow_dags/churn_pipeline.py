from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction pipeline',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['mlops', 'churn'],
)

# Define tasks
# Using BashOperator to run the python scripts. 
# In a real production setup, we might use DockerOperator or KubernetesPodOperator.
# Assuming airflow is running where python environment is set up.

t1_ingest = BashOperator(
    task_id='data_ingestion',
    bash_command='python src/data_ingestion.py --output data/raw/raw_data.csv',
    dag=dag,
)

t2_preprocess = BashOperator(
    task_id='data_preprocessing',
    bash_command='python src/preprocessing.py --input data/raw/raw_data.csv --output data/processed',
    dag=dag,
)

t3_train = BashOperator(
    task_id='model_training',
    bash_command='python src/train.py --data data/processed/train.csv --model-dir models',
    dag=dag,
)

t4_evaluate = BashOperator(
    task_id='model_evaluation',
    bash_command='python src/evaluate.py --model-dir models --test-data data/processed/test.csv --metrics-file metrics.json',
    dag=dag,
)

# Define dependency chain
t1_ingest >> t2_preprocess >> t3_train >> t4_evaluate
