#!/bin/bash

# Активация conda-окружения
source /home/miniconda3/etc/profile.d/conda.sh
conda activate image_rag  # Убедитесь, что указали правильное окружение

# Установим переменные окружения для базы данных и путей
export IMAGE_DIR="resources/images"
export DB_PATH="resources/images_metadata.db"
export MODEL_DIR='/home/meno/image_rag/Image-Rag/ONE-PEACE'
export MODEL_NAME='/home/meno/models/one-peace.pt'

# Запуск приложения с логированием
nohup python3 backend.py > logs/backend_output.log 2> logs/backend_error.log
