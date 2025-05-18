# Домашнее задание 2: Погодный API с AirFlow

## Описание проекта

Этот проект демонстрирует использование Apache Airflow для регулярного сбора данных о погоде в Москве через API сервиса OpenWeatherMap.

DAG в Airflow настроен на запуск каждую минуту и сохраняет данные о погоде в CSV файл.

## Структура проекта

- `weather.py` - модуль для работы с API OpenWeatherMap и сохранения данных
- `weather_dag.py` - DAG для Airflow
- `test_weather.py` - скрипт для тестирования без Airflow
- `.env` - файл с API ключом (не отслеживается в Git)
- `requirements.txt` - необходимые зависимости

## Проверка работоспособности

### Метод 1: Без использования Airflow

1. Зарегистрируйтесь на [OpenWeatherMap](https://openweathermap.org/) и получите API ключ

2. Укажите ваш API ключ в файле `.env`:
```
OPENWEATHERMAP_API_KEY=your_api_key_here
```

3. Создайте и активируйте виртуальное окружение:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

4. Установите зависимости:
```bash
pip install -r requirements.txt
```

5. Запустите тестовый скрипт:
```bash
python test_weather.py
```

6. Проверьте результаты в файле `weather.csv`. Скрипт выполнит 3 запроса с интервалом 10 секунд.

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

2. Разместите DAG в директории dags Airflow и убедитесь, что файл `.env` доступен для Airflow:
```bash
mkdir -p ~/airflow/dags
cp weather_dag.py ~/airflow/dags/
cp .env ~/airflow/
```

3. Запустите веб-сервер и планировщик Airflow:
```bash
# Терминал 1
airflow webserver --port 8080

# Терминал 2
airflow scheduler
```

4. Откройте веб-интерфейс Airflow в браузере: http://localhost:8080

5. Активируйте DAG 'weather_data_pipeline' через интерфейс и наблюдайте за его выполнением. 
   Каждую минуту должна появляться новая запись в файле `weather.csv`. 