import sys
import os

import pytesseract
import torch
import sqlite3
import numpy as np
import logging
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from scipy.spatial.distance import cdist
from deep_translator import GoogleTranslator
from deepface import DeepFace
from typing import Optional

IMAGE_DIR = os.path.abspath("/home/meno/image_rag/Image-RAG/resources/val2017")
DB_PATH = os.path.abspath("/home/meno/image_rag/Image-RAG/resources/images_metadata.db")
MODEL_DIR = 'ONE-PEACE/'
MODEL_NAME = '/home/meno/models/one-peace.pt'

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Настройка pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Укажите путь к исполняемому файлу tesseract


def translate_to_english(text):
    translator = GoogleTranslator(source='ru', target='en')
    translated_text = translator.translate(text)
    return translated_text


def recognize_celebrities(image: Image.Image):
    """Распознавание всех знаменитостей на изображении с помощью DeepFace."""
    try:
        # Преобразуем изображение в массив numpy
        image_np = np.array(image)

        # Анализ всех лиц на изображении
        results = DeepFace.analyze(image_np, actions=['identity'], detector_backend='opencv', enforce_detection=False)

        # Если результат содержит список лиц
        if isinstance(results, list):
            # Возвращаем список идентифицированных знаменитостей для каждого лица
            return [result.get('identity', 'Unknown') for result in results]
        else:
            return [results.get('identity', 'Unknown')]
    except Exception as e:
        print(f"Ошибка распознавания лиц: {e}")
        return []


def setup_models(model_dir=MODEL_DIR, model_name=MODEL_NAME):
    """Загрузка модели ONE-PEACE с проверкой путей."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'The directory "{model_dir}" does not exist')
    if not os.path.isfile(model_name):
        raise FileNotFoundError(f'The model file "{model_name}" does not exist')

    one_peace_dir = os.path.normpath(MODEL_DIR)
    if not os.path.isdir(one_peace_dir):
        err_msg = f'The dir "{one_peace_dir}" does not exist'
        logger.error(err_msg)
        raise ValueError(err_msg)

    model_name = os.path.normpath(MODEL_NAME)
    if not os.path.isfile(model_name):
        err_msg = f'The file "{model_name}" does not exist'
        logger.error(err_msg)
        raise ValueError(err_msg)
    sys.path.append(one_peace_dir)
    from one_peace.models import from_pretrained

    logger.info("Загрузка модели ONE-PEACE")
    current_workdir = os.getcwd()
    logger.info(f'Текущая рабочая директория: {current_workdir}')

    os.chdir(one_peace_dir)
    logger.info(f'Новая рабочая директория: {os.getcwd()}')
    model = from_pretrained(model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    logger.info("Загрузка модели SBERT")
    model_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model, model_sbert


def initialize_database(db_path):
    """Создание базы данных, если она не существует."""
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_name TEXT PRIMARY KEY,
            op_embedding BLOB,
            recognized_text TEXT,
            recognized_text_embedding BLOB,
            celebrity_names TEXT,
            text_description TEXT,
            text_description_embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()


def save_embedding(db_path, image_name, embedding, recognized_text, celebrity_names, text_description=None,
                   text_description_embedding=None):
    """Сохранение эмбеддингов и текста OCR в базу данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO image_embeddings (image_name, op_embedding, recognized_text, celebrity_names, text_description, text_description_embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        image_name,
        embedding.tobytes(),
        recognized_text,
        ", ".join(celebrity_names),
        text_description,
        text_description_embedding.tobytes() if text_description_embedding is not None else None

    ))
    conn.commit()
    conn.close()


def get_image_embeddings(db_path):
    """Получение всех эмбеддингов изображений и текстов OCR из базы данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT image_name, op_embedding, recognized_text FROM image_embeddings')
    results = cursor.fetchall()
    conn.close()

    image_names = [row[0] for row in results]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in results]
    ocr_texts = [row[2] for row in results]

    return image_names, np.vstack(embeddings), ocr_texts


def vectorize_image(model, transform, image: Image.Image, device):
    """Векторизация изображения с помощью модели."""
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.extract_image_features(image).cpu().numpy()
    return embedding


def create_transforms():
    """Создание трансформаций для изображений."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера."""
    global model_op, model_sbert, transform_op
    initialize_database(DB_PATH)
    model_op, model_sbert = setup_models()
    transform_op = create_transforms()
    logger.info("Сервер запущен и готов к работе.")


def get_next_image_number(image_dir):
    """Функция для получения следующего номера изображения на основе существующих файлов."""
    existing_files = os.listdir(image_dir)
    if not existing_files:
        return 0

    image_numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
    if not image_numbers:
        return 0

    return max(image_numbers) + 1


def get_image_embeddings_with_celebrity(db_path):
    """Получение всех эмбеддингов изображений, текстов OCR, описаний и имен знаменитостей из базы данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT image_name, op_embedding, recognized_text, celebrity_names, text_description_embedding FROM image_embeddings')
    results = cursor.fetchall()
    conn.close()

    image_names = [row[0] for row in results]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in results]
    ocr_texts = [row[2] for row in results]
    celebrity_names = [row[3] for row in results]
    text_description_embeddings = [np.frombuffer(row[4], dtype=np.float32) if row[4] is not None else None for row in
                                   results]

    return image_names, np.vstack(embeddings), ocr_texts, celebrity_names, text_description_embeddings


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), description: Optional[str] = None):
    """Загрузка изображения и сохранение его эмбеддингов."""
    logger.info("Изображение получено.")
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=400, detail="Невозможно обработать изображение")

    embedding = vectorize_image(model_op, transform_op, image,
                                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Использование pytesseract для распознавания текста
    ocr_result = pytesseract.image_to_string(image, lang='eng+rus')
    ocr_text = ocr_result.strip()

    celebrity_names = recognize_celebrities(image)
    logger.info(f"Знаменитости, распознанные на изображении: {', '.join(celebrity_names)}")

    description_embedding = None
    if description:
        description_embedding = model_sbert.encode(description)

    next_image_number = get_next_image_number(IMAGE_DIR)
    new_image_name = f"{next_image_number:05d}.jpg"
    image_path = os.path.join(IMAGE_DIR, new_image_name)

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    image.save(image_path)
    save_embedding(DB_PATH, new_image_name, embedding, ocr_text, celebrity_names, description, description_embedding)

    logger.info(f"Изображение '{image_path}' загружено и обработано.")
    return JSONResponse({"status": "Отправленное изображение сохранено в общую базу данных и обработано."})


class QueryRequest(BaseModel):
    query: str


@app.post("/search/")
async def search_images(query: QueryRequest):
    """Поиск изображений по запросу."""
    logger.info(f"Текстовый запрос получен: {query.query}")
    translated_query = translate_to_english(query.query)
    logger.info(f"Запрос переведен на английский: {translated_query}")

    text_tokens = model_op.process_text([translated_query])
    with torch.no_grad():
        text_features = model_op.extract_text_features(text_tokens).cpu().numpy()

    query_text_embedding = model_sbert.encode(query.query)

    # Получаем эмбеддинги изображений, OCR текстов и имен знаменитостей
    image_names, image_embeddings, ocr_texts, celebrity_names, text_description_embeddings = get_image_embeddings_with_celebrity(
        DB_PATH)

    # Рассчитываем расстояния между запросом и эмбеддингами изображений
    distances_one_peace = cdist(text_features, image_embeddings, metric='cosine').flatten()

    # Рассчитываем расстояния между запросом и текстами OCR
    ocr_embeddings = model_sbert.encode(ocr_texts)
    query_text_embedding = np.array(query_text_embedding).reshape(1, -1)
    ocr_embeddings = np.array(ocr_embeddings)
    distances_ocr = cdist(query_text_embedding, ocr_embeddings, metric='cosine').flatten()

    # Рассчитываем расстояния между запросом и именами знаменитостей
    celebrity_embeddings = model_sbert.encode(celebrity_names)
    distances_celebrities = cdist(query_text_embedding, celebrity_embeddings, metric='cosine').flatten()

    if text_description_embeddings:
        text_description_embeddings = np.array(
            embedding for embedding in text_description_embeddings if embedding is not None)
        distances_descriptions = cdist(query_text_embedding, text_description_embeddings,
                                       metric='cosine').flatten()
    else:
        distances_descriptions = np.zeros(len(image_names))

    # Комбинируем расстояния (по изображениям, текстам и знаменитостям)
    combined_distances = (distances_one_peace + distances_ocr + distances_descriptions) / 3
    indices = np.argsort(combined_distances)[:10]

    # Формируем результаты поиска
    results = [{"image_name": image_names[i], "similarity": 1 - combined_distances[i]} for i in indices]
    logger.info(f"Поиск по запросу '{query.query}' завершен. Найдено {len(results)} результатов.")
    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
