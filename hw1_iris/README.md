# Домашнее задание 1: Пайплайн ML с AirFlow

## Описание проекта

Этот проект демонстрирует использование Apache Airflow для оркестрации пайплайна машинного обучения на примере классификатора на наборе данных "Ирисы Фишера".

Пайплайн состоит из следующих этапов:
1. Загрузка данных
2. Подготовка данных (разделение на обучающую и тестовую выборки)
3. Обучение модели (логистическая регрессия)
4. Тестирование модели

## Структура проекта

- `data.py` - функции для загрузки и подготовки данных
- `train.py` - функция обучения модели
- `test.py` - функция тестирования модели
- `iris_dag.py` - DAG для Airflow
- `run_pipeline.py` - скрипт для запуска пайплайна без Airflow
- `requirements.txt` - необходимые зависимости

## Проверка работоспособности

### Метод 1: Без использования Airflow

1. Создайте и активируйте виртуальное окружение:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

2. Установите зависимости в правильном порядке:
```bash
pip install numpy==1.24.3
pip install -r requirements.txt
```

3. Запустите скрипт `run_pipeline.py`:
```bash
python run_pipeline.py
```

4. Проверьте результаты:
   - Датасет: `dataset/iris.csv`
   - Тренировочная выборка: `dataset/iris_train.csv`
   - Тестовая выборка: `dataset/iris_test.csv`
   - Модель: `model.pkl`
   - Метрики: `model_metrics.json`

### Метод 2: С использованием Airflow

1. Установите и настройте Airflow:
```bash
# Инициализация базы данных
airflow db init

# Создание пользователя для веб-интерфейса
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

2. Разместите DAG в директории dags Airflow:
```bash
mkdir -p ~/airflow/dags
cp iris_dag.py ~/airflow/dags/
```

3. Запустите веб-сервер и планировщик Airflow:
```bash
# Терминал 1
airflow webserver --port 8080

# Терминал 2
airflow scheduler
```

4. Откройте веб-интерфейс Airflow в браузере: http://localhost:8080

5. Активируйте DAG 'iris_ml_pipeline' через интерфейс и запустите его вручную. 