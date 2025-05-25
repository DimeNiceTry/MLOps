import os
import requests
import pandas as pd
import csv
import datetime
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения из .env файла
load_dotenv()

def fetch_weather(city="Moscow"):
    """
    Получает данные о погоде для указанного города.
    
    Args:
        city (str): Город, для которого нужно получить данные о погоде. По умолчанию - Москва.
        
    Returns:
        dict: Словарь с данными о погоде
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise ValueError("API ключ не найден. Убедитесь, что в файле .env есть OPENWEATHERMAP_API_KEY")
    
    # URL для API запроса
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=ru"
    
    # Отправляем запрос
    response = requests.get(url)
    
    # Проверяем успешность запроса
    if response.status_code != 200:
        raise Exception(f"Ошибка API запроса: {response.status_code}, {response.text}")
    
    # Возвращаем данные в формате JSON
    return response.json()

def save_weather_data(city="Moscow"):
    """
    Получает данные о погоде и сохраняет их в CSV файл.
    
    Args:
        city (str): Город, для которого нужно получить данные о погоде. По умолчанию - Москва.
        
    Returns:
        str: Путь к CSV файлу с данными
    """
    # Получаем данные о погоде
    weather_data = fetch_weather(city)
    
    # Извлекаем нужные данные
    data = {
        'datetime': datetime.datetime.fromtimestamp(weather_data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
        'city': city,
        'weather_main': weather_data['weather'][0]['main'],
        'weather_description': weather_data['weather'][0]['description'],
        'temp': weather_data['main']['temp'],
        'feels_like': weather_data['main']['feels_like'],
        'pressure': weather_data['main']['pressure'],
        'wind_speed': weather_data['wind']['speed']
    }
    
    # Путь к CSV файлу
    csv_path = 'weather.csv'
    
    # Проверяем, существует ли файл
    file_exists = Path(csv_path).exists()
    
    # Открываем файл для добавления данных
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        
        # Если файл не существует, записываем заголовки
        if not file_exists:
            writer.writeheader()
        
        # Записываем данные
        writer.writerow(data)
    
    print(f"Данные о погоде сохранены в {csv_path}")
    return csv_path

if __name__ == "__main__":
    # Для тестирования
    save_weather_data() 