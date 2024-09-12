import sys
import os
import asyncio
import logging
from collections import defaultdict

from aiogram import Bot, Dispatcher, types, BaseMiddleware
from aiogram.filters import Command
from aiogram import F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InputMediaPhoto, Message
from dotenv import load_dotenv
import aiohttp


class ThrottlingMiddleware(BaseMiddleware):
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.last_time = defaultdict(lambda: 0)
        super().__init__()

    async def __call__(self, handler, event: Message, data):
        # Проверяем, является ли событие сообщением и содержит ли оно from_user
        if not isinstance(event, Message) or not event.from_user:
            logger.warning("Получено событие без from_user.")
            return await handler(event, data)

        current_time = asyncio.get_event_loop().time()
        user_id = event.from_user.id

        # Проверяем разницу между последним запросом и текущим временем
        if current_time - self.last_time[user_id] < self.rate_limit:
            logger.info(f"Rate limit exceeded for user {user_id}")
            await event.answer("Слишком много запросов. Подождите немного.")
            return
        else:
            # Обновляем время последнего запроса пользователя
            self.last_time[user_id] = current_time

        return await handler(event, data)


# Настройка логирования
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
dp.update.middleware(ThrottlingMiddleware(rate_limit=5))

# Установка лимитов на запросы
RATE_LIMIT = 5  # seconds


# Функция для получения изображений асинхронно
async def fetch_image(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            logger.error(f"Ошибка при скачивании изображения {url}: {response.status}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при скачивании изображения: {e}")
        return None


# Обработка неподдерживаемых форматов
@dp.message(F.document | F.sticker | F.audio | F.video)
async def handle_unsupported_content(message: types.Message):
    if not message.from_user:
        logger.warning("Получено некорректное сообщение без пользователя.")
        return
    logger.info(f"Неподдерживаемый формат от пользователя {message.from_user.id}")
    await message.answer("Этот формат не поддерживается. Отправьте текст или изображение.")


# Обработчик команды /start
@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    logger.info(f"Пользователь {message.from_user.id} отправил команду /start")
    await message.answer("Привет! Отправь изображение для загрузки или текст для поиска.")


# Ограничение частоты запросов
async def anti_flood(*args, **kwargs):
    message = args[0]
    await message.answer("Слишком много запросов. Подождите немного.")


# Обработка изображений
@dp.message(F.photo)
async def handle_image(message: types.Message):
    if not message.from_user:
        logger.warning("Получено сообщение без пользователя.")
        return

    logger.info(f"Получено изображение от пользователя {message.from_user.id}")
    try:
        photo = message.photo[-1]  # Берём фото с наибольшим разрешением
        file_info = await bot.get_file(photo.file_id)
        file_path = file_info.file_path

        # Асинхронное скачивание изображения
        image_url = f'https://api.telegram.org/file/bot{API_TOKEN}/{file_path}'
        async with aiohttp.ClientSession() as session:
            image_content = await fetch_image(session, image_url)
            if not image_content:
                await message.answer("Не удалось загрузить изображение.")
                return

        # Асинхронная отправка изображения на бэкенд
        files = {'file': ('image.jpg', image_content)}
        async with session.post(f"{API_URL}/upload-image/", data=files) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"Изображение успешно загружено на сервер: {file_path}")
                await message.answer(result["status"])
            else:
                logger.error(f"Ошибка при отправке изображения на бэкенд: {response.status}")
                await message.answer("Ошибка при загрузке изображения.")
    except Exception as e:
        logger.exception("Произошла ошибка при обработке изображения.")
        await message.answer("Произошла ошибка при обработке изображения.")


# Обработка текстовых запросов
@dp.message(F.text)
async def handle_text(message: types.Message):
    if not message.from_user:
        logger.warning("Получено сообщение без пользователя.")
        return
    logger.info(f"Получен текстовый запрос от пользователя {message.from_user.id}: {message.text}")
    try:
        query = message.text
        logger.info(f"Отправка запроса на поиск: {query}")

        # Асинхронная отправка запроса на поиск
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_URL}/search/", json={"query": query}) as response:
                if response.status == 200:
                    results = await response.json()
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
                    logger.error(f"Ошибка при запросе поиска: {response.status}")
                    await message.answer(f"Ошибка при поиске: {response.status}")
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
