import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression


def train(train_csv: str) -> str:
    """Обучение модели логистической регрессии на тренировочной выборке и сохранение модели."""
    # Загружаем обучающую выборку
    df = pd.read_csv(train_csv)
    
    # Разделяем признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Создаем и обучаем модель логистической регрессии
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Сохраняем модель в файл
    model_path = 'model.pkl'
    joblib.dump(model, model_path)
    
    return model_path
