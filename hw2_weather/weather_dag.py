from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from weather import save_weather_data

# Определяем аргументы DAG по умолчанию
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'start_date': datetime(2023, 1, 1),
}

# Создаем DAG
dag = DAG(
    'weather_data_pipeline',
    default_args=default_args,
    description='Сбор данных о погоде в Москве каждую минуту',
    schedule_interval=timedelta(minutes=1),  # Запуск каждую минуту
    catchup=False,
)

# Задача для сбора данных о погоде
fetch_weather_task = PythonOperator(
    task_id='fetch_weather_task',
    python_callable=save_weather_data,
    op_kwargs={'city': 'Moscow'},
    dag=dag,
)

# В этом DAG есть только одна задача, поэтому не требуется определять последовательность выполнения 