from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

IMAGE_DIR = "./images"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Логируем запрос с изображением
    logger.info(f"Получен файл: {file.filename}")

    # Сохраняем изображение без обработки
    image_path = os.path.join(IMAGE_DIR, file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Логируем успешное сохранение файла
    logger.info(f"Изображение сохранено: {image_path}")

    # Возвращаем ответ без обработки моделей
    return JSONResponse({"status": "Изображение загружено, но не обработано."})


class QueryRequest(BaseModel):
    query: str


@app.post("/search/")
async def search_images(query: QueryRequest):
    # Логируем текстовый запрос
    logger.info(f"Получен текстовый запрос: {query.query}")

    # Возвращаем пустой ответ, так как пока не обрабатываем запросы
    return JSONResponse({"status": "Поиск временно не работает. Логирование выполнено."})


def main():
    # Запуск сервера с использованием Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
