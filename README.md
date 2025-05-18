# Домашние задания по MLOps

В этом репозитории содержатся два домашних задания по теме "Оркестраторы" с использованием Apache Airflow:

1. `hw1_iris` - Пайплайн машинного обучения для классификации ирисов
2. `hw2_weather` - Сбор данных о погоде через API OpenWeatherMap

## Инструкция по проверке работоспособности

### Домашнее задание 1: Пайплайн ML для классификации ирисов

#### Быстрая проверка без Airflow:

1. Перейдите в директорию первого задания:
```bash
cd hw1_iris
```

2. Создайте и активируйте виртуальное окружение:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Установите зависимости в правильном порядке:
```bash
pip install numpy==1.24.3
pip install -r requirements.txt
```

4. Запустите пайплайн:
```bash
python run_pipeline.py
```

5. После успешного запуска должны появиться следующие файлы:
   - `dataset/iris.csv` - исходный датасет
   - `dataset/iris_train.csv` - обучающая выборка
   - `dataset/iris_test.csv` - тестовая выборка
   - `model.pkl` - обученная модель
   - `model_metrics.json` - метрики модели

Полные инструкции по настройке Airflow находятся в [README.md](hw1_iris/README.md) первого задания.

### Домашнее задание 2: Сбор данных о погоде

#### Быстрая проверка без Airflow:

1. Перейдите в директорию второго задания:
```bash
cd hw2_weather
```

2. Убедитесь, что в файле `.env` указан ваш API ключ OpenWeatherMap:
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

6. После успешного запуска должен появиться файл `weather.csv` с данными о погоде.

Полные инструкции по настройке Airflow находятся в [README.md](hw2_weather/README.md) второго задания.

## Примечания

- В обоих заданиях предусмотрена возможность запуска без настройки Airflow для быстрой проверки функциональности.
- В файлах `.gitignore` указаны файлы, которые не должны попадать в репозиторий.
- API ключ OpenWeatherMap не должен быть загружен в репозиторий и хранится в файле `.env`, который указан в `.gitignore`.
