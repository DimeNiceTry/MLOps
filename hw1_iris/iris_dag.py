from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from data import load_data, prepare_data
from train import train
from test import test

# Определяем аргументы DAG по умолчанию
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 1, 1),
}

# Создаем DAG
dag = DAG(
    'iris_ml_pipeline',
    default_args=default_args,
    description='Пайплайн для обучения и тестирования классификатора на датасете Iris',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Определяем функции для связи этапов
def load_data_task():
    return load_data()

def prepare_data_task(ti):
    csv_path = ti.xcom_pull(task_ids='load_data_task')
    return prepare_data(csv_path)

def train_model_task(ti):
    train_test_paths = ti.xcom_pull(task_ids='prepare_data_task')
    train_path = train_test_paths[0]  # Первый элемент - путь к обучающей выборке
    return train(train_path)

def test_model_task(ti):
    model_path = ti.xcom_pull(task_ids='train_model_task')
    train_test_paths = ti.xcom_pull(task_ids='prepare_data_task')
    test_path = train_test_paths[1]  # Второй элемент - путь к тестовой выборке
    return test(model_path, test_path)

# Создаем задачи (tasks)
task_load_data = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data_task,
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id='prepare_data_task',
    python_callable=prepare_data_task,
    dag=dag,
)

task_train_model = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model_task,
    dag=dag,
)

task_test_model = PythonOperator(
    task_id='test_model_task',
    python_callable=test_model_task,
    dag=dag,
)

# Определяем порядок выполнения задач
task_load_data >> task_prepare_data >> task_train_model >> task_test_model 