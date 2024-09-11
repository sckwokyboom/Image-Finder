#!/bin/bash

source /home/miniconda/etc/profile.d/conda.sh

conda activate image_rag

# Запуск Telegram-бота в фоновом режиме с логированием
nohup python3 src/front/telegram_bot.py > logs/telegram_bot.log 2>&1 &

echo "Telegram-бот запущен в фоне через conda окружение my_bot_env и логируется в telegram_bot.log"
