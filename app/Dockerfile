# Используем официальный образ Python с поддержкой нужной версии (например, 3.9)
FROM python:3.9-slim

# Опционально: установка системных зависимостей (например, для работы с изображениями)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем весь исходный код в контейнер
COPY . .

# Открываем порт 5000 (порт Flask по умолчанию)
EXPOSE 5000

# Опционально: если хотите запускать через gunicorn для production
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# Для простоты можно запускать приложение напрямую:
CMD ["python", "app.py"]
