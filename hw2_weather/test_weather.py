import time
from weather import save_weather_data

def test_weather_collection(count=3, interval=60):
    """
    Тестирует сбор данных о погоде несколько раз с заданным интервалом
    
    Args:
        count (int): Количество раз для сбора данных
        interval (int): Интервал между сборами в секундах (по умолчанию 60 секунд)
    """
    print(f"Запуск тестирования сбора данных о погоде. Будет выполнено {count} запросов с интервалом {interval} секунд")
    
    for i in range(count):
        print(f"\nЗапрос {i+1}/{count}")
        try:
            csv_path = save_weather_data()
            print(f"Данные успешно сохранены в {csv_path}")
            
            # Если это не последний запрос, ждем заданный интервал
            if i < count - 1:
                print(f"Ожидание {interval} секунд до следующего запроса...")
                time.sleep(interval)
        except Exception as e:
            print(f"Ошибка при сборе данных: {e}")
            break
    
    print("\nТестирование завершено. Проверьте файл weather.csv для просмотра собранных данных.")

if __name__ == "__main__":
    # Запускаем тестирование с 3 запросами и интервалом 10 секунд (для быстрого тестирования)
    test_weather_collection(count=3, interval=10) 