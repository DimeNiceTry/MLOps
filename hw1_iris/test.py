import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report


def test(model_path: str, test_csv: str) -> str:
    """Тестирование модели на тестовой выборке и сохранение результатов."""
    # Загружаем тестовую выборку
    df = pd.read_csv(test_csv)
    
    # Разделяем признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Загружаем модель
    model = joblib.load(model_path)
    
    # Предсказываем на тестовой выборке
    y_pred = model.predict(X)
    
    # Вычисляем метрики
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    
    # Сохраняем метрики в json файл
    metrics_path = 'model_metrics.json'
    metrics = {'accuracy': accuracy, 'report': report}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    return metrics_path
