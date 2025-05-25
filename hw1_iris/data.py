import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data() -> str:
    """Загрузка датасета Iris."""
    # Создаем директорию для датасета, если ее нет
    os.makedirs("dataset", exist_ok=True)
    
    # Загружаем датасет Iris из sklearn
    iris = load_iris()
    
    # Создаем DataFrame из данных
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Сохраняем датасет в csv
    csv_path = "dataset/iris.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


def prepare_data(csv_path: str) -> list[str]:
    """Чтение загруженного датасета и разделение на train и test выборки."""
    # Загружаем датасет
    df = pd.read_csv(csv_path)
    
    # Разделяем признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Разделяем на обучающую и тестовую выборки (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Создаем DataFrame для обучающей и тестовой выборок
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Пути для сохранения выборок
    train_csv = "dataset/iris_train.csv"
    test_csv = "dataset/iris_test.csv"
    
    # Сохраняем выборки в csv
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    return [train_csv, test_csv]
