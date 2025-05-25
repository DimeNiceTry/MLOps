from data import load_data, prepare_data
from train import train
from test import test

def run_pipeline():
    # Шаг 1: Загрузка данных
    print("Шаг 1: Загрузка данных...")
    csv_path = load_data()
    print(f"Данные загружены в {csv_path}")
    
    # Шаг 2: Подготовка данных
    print("\nШаг 2: Подготовка данных...")
    train_test_paths = prepare_data(csv_path)
    print(f"Данные разделены: {train_test_paths}")
    
    # Шаг 3: Обучение модели
    print("\nШаг 3: Обучение модели...")
    model_path = train(train_test_paths[0])
    print(f"Модель обучена и сохранена в {model_path}")
    
    # Шаг 4: Тестирование модели
    print("\nШаг 4: Тестирование модели...")
    metrics_path = test(model_path, train_test_paths[1])
    print(f"Модель протестирована, метрики сохранены в {metrics_path}")
    
    print("\nВесь пайплайн успешно выполнен!")

if __name__ == "__main__":
    run_pipeline() 