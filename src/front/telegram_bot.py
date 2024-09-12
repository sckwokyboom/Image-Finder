import sys
import requests
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram import F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InputMediaPhoto
from dotenv import load_dotenv
import os

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
API_URL = os.getenv("API_URL")

# Проверка наличия необходимых переменных
if not API_TOKEN or not API_URL:
    logger.error("Не удалось загрузить API_TOKEN или API_URL из .env файла")
    raise ValueError("Проверьте файл .env для корректной настройки токена и URL")

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


# Стартовая команда
@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    logger.info(f"Пользователь {message.from_user.id} отправил команду /start")
    await message.answer("Привет! Отправь изображение для загрузки или текст для поиска.")


# Обработка изображений
@dp.message(F.photo)
async def handle_image(message: types.Message):
    logger.info(f"Получено изображение от пользователя {message.from_user.id}")
    try:
        photo = message.photo[-1]  # Берём фото с наибольшим разрешением
        file_info = await bot.get_file(photo.file_id)
        file_path = file_info.file_path

        # Получаем URL для скачивания изображения
        image_url = f'https://api.telegram.org/file/bot{API_TOKEN}/{file_path}'
        image_response = requests.get(image_url)

        # Проверка успешности скачивания
        if image_response.status_code != 200:
            logger.error(f"Ошибка при загрузке изображения {file_path}")
            await message.answer("Не удалось загрузить изображение.")
            return

        files = {'file': ('image.jpg', image_response.content)}

        # Отправляем изображение на бэкенд
        response = requests.post(f"{API_URL}/upload-image/", files=files)
        if response.status_code == 200:
            logger.info(f"Изображение успешно загружено на сервер: {file_path}")
            await message.answer(response.json()["status"])
        else:
            logger.error(f"Ошибка при отправке изображения на бэкенд: {response.status_code}")
            await message.answer("Ошибка при загрузке изображения.")
    except Exception as e:
        logger.exception("Произошла ошибка при обработке изображения.")
        await message.answer("Произошла ошибка при обработке изображения.")


@dp.message(F.text)
async def handle_text(message: types.Message):
    logger.info(f"Получен текстовый запрос от пользователя {message.from_user.id}: {message.text}")
    try:
        query = message.text
        logger.info(f"Отправка запроса на поиск: {query}")
        response = requests.post(f"{API_URL}/search/", json={"query": query})

        if response.status_code == 200:
            logger.info(f"Успешный ответ от бэкенда: {response.status_code}")
            results = response.json()
            if results:
                media_group = []
                logger.info(f"Найдено {len(results)} результатов, обработка первых 10")

                for result in results[:10]:
                    image_name = result['image_name']
                    similarity = result['similarity']
                    image_path = os.path.join("/home/meno/image_rag/Image-RAG/resources/val2017/", image_name)

                    logger.info(f"Проверка существования файла: {image_path}")
                    if os.path.exists(image_path):
                        try:
                            input_file = types.FSInputFile(image_path)

                            media_group.append(InputMediaPhoto(
                                media=input_file,
                                caption=f"Похожесть: {similarity:.4f}"
                            ))
                        except Exception as e:
                            logger.error(f"Ошибка при создании InputMediaPhoto для {image_path}: {e}")
                    else:
                        logger.error(f"Файл не найден: {image_path}")

                if media_group:
                    logger.info(f"Отправка {len(media_group)} изображений в виде группы")
                    try:
                        await message.answer_media_group(media_group)
                    except Exception as e:
                        logger.error(f"Ошибка при отправке группы изображений: {e}")
                        await message.answer("Ошибка при отправке изображений.")
                else:
                    logger.warning("Медиа-группа пуста, нет изображений для отправки")
                    await message.answer("Не удалось найти изображения.")
            else:
                logger.info("По запросу ничего не найдено.")
                await message.answer("Ничего не найдено.")
        else:
            logger.error(f"Ошибка при запросе поиска: {response.status_code}")
            await message.answer(f"Ошибка при поиске: {response.status_code}")
    except Exception as e:
        logger.exception(f"Произошла ошибка при обработке текстового запроса: {e}")
        await message.answer("Произошла ошибка при обработке запроса.")


# Основная функция
async def main():
    logger.info("Бот запускается...")
    try:
        await dp.start_polling(bot)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")


if __name__ == '__main__':
    asyncio.run(main())
