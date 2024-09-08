import requests
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType
from aiogram.filters import Command
from aiogram import F
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
API_URL = os.getenv("API_URL")

if not API_TOKEN or not API_URL:
    logger.error("Не удалось загрузить API_TOKEN или API_URL из .env файла")
    raise ValueError("Проверьте файл .env для корректной настройки токена и URL")

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    logger.info(f"Пользователь {message.from_user.id} отправил команду /start")
    await message.answer("Привет! Отправь изображение для загрузки или текст для поиска.")


@dp.message(F.photo)
async def handle_image(message: types.Message):
    logger.info(f"Получено изображение от пользователя {message.from_user.id}")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        file_path = file_info.file_path

        image_url = f'https://api.telegram.org/file/bot{API_TOKEN}/{file_path}'
        image_response = requests.get(image_url)

        if image_response.status_code != 200:
            logger.error(f"Ошибка при загрузке изображения {file_path}")
            await message.answer("Не удалось загрузить изображение.")
            return

        files = {'file': image_response.content}

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
        response = requests.post(f"{API_URL}/search/", json={"query": query})

        if response.status_code == 200:
            results = response.json()
            if results:
                for result in results:
                    await message.answer(f"Image: {result['image_name']}, Similarity: {result['similarity']:.4f}")
            else:
                await message.answer("Ничего не найдено.")
        else:
            logger.error(f"Ошибка при запросе поиска: {response.status_code}")
            await message.answer("Ошибка при поиске.")
    except Exception as e:
        logger.exception("Произошла ошибка при обработке текстового запроса.")
        await message.answer("Произошла ошибка при обработке запроса.")


async def main():
    logger.info("Бот запускается...")
    try:
        await dp.start_polling(bot)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")


if __name__ == '__main__':
    asyncio.run(main())
