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
from fastapi import Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from scipy.spatial.distance import cdist
from deep_translator import GoogleTranslator
from typing import Optional
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

IMAGE_DIR = os.path.abspath("/home/meno/image_rag/Image-RAG/resources/val2017")
DB_PATH = os.path.abspath("/home/meno/image_rag/Image-RAG/resources/images_metadata.db")
MODEL_DIR = 'ONE-PEACE/'
MODEL_NAME = '/home/meno/models/one-peace.pt'

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
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


def setup_models(model_dir=MODEL_DIR, model_name=MODEL_NAME):
    """Загрузка модели ONE-PEACE с проверкой путей."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'The directory "{model_dir}" does not exist')
    if not os.path.isfile(model_name):
        raise FileNotFoundError(f'The model file "{model_name}" does not exist')

    nltk.download('punkt', '/home/meno/models/nltk_data')
    nltk.data.path.append('/home/meno/models/nltk_data')
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
    logger.info("ONE-PEACE был успешно загружен")
    logger.info("Загрузка модели SBERT")
    model_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("SBERT был успешно загружен")
    return model, model_sbert


def initialize_database(db_path):
    """Создание базы данных, если она не существует."""
    logger.info("Инициализация базы данных...")
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
            text_description TEXT,
            text_description_embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Инициализация базы данных завершена")


def save_embedding(db_path, image_name, embedding, recognized_text, text_description=None,
                   text_description_embedding=None):
    """Сохранение эмбеддингов и текста OCR в базу данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO image_embeddings (image_name, op_embedding, recognized_text, text_description, text_description_embedding)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        image_name,
        embedding.tobytes(),
        recognized_text,
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


def get_image_embeddings(db_path):
    """Получение всех эмбеддингов изображений, текстов OCR, описаний и имен знаменитостей из базы данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT image_name, op_embedding, recognized_text, text_description_embedding FROM image_embeddings')
    results = cursor.fetchall()
    conn.close()

    image_names = [row[0] for row in results]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in results]
    # TODO: очень плохо с индексами тут работать, явно неправильно
    ocr_texts = [row[2] for row in results]
    text_description_embeddings = [np.frombuffer(row[3], dtype=np.float32) if row[3] is not None else None for row in
                                   results]

    return image_names, np.vstack(embeddings), ocr_texts, text_description_embeddings


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), description: Optional[str] = Form(None)):
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
    if ocr_text:
        logging.info(f"Распознанный текст: {ocr_text}")
    else:
        logging.warning("Не удалось распознать текст на изображении")

    description_embedding = None
    if description:
        logging.info(f"Получено описание изображения: {description}")
        description_embedding = model_sbert.encode(description)
    else:
        logging.warning("Описание изображения отсутствует")

    next_image_number = get_next_image_number(IMAGE_DIR)
    new_image_name = f"{next_image_number:05d}.jpg"
    image_path = os.path.join(IMAGE_DIR, new_image_name)

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    image.save(image_path)
    save_embedding(DB_PATH, new_image_name, embedding, ocr_text, description, description_embedding)

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
    image_names, image_embeddings, ocr_texts, text_description_embeddings = get_image_embeddings(
        DB_PATH)

    # Рассчитываем расстояния между запросом и эмбеддингами изображений
    distances_one_peace = cdist(text_features, image_embeddings, metric='cosine').flatten()

    # Рассчитываем расстояния между запросом и текстами OCR
    ocr_embeddings = model_sbert.encode(ocr_texts)
    query_text_embedding = np.array(query_text_embedding).reshape(1, -1)
    ocr_embeddings = np.array(ocr_embeddings)
    distances_ocr = cdist(query_text_embedding, ocr_embeddings, metric='cosine').flatten()

    distances_descriptions = np.ones(len(image_names))

    if text_description_embeddings:
        # Подготавливаем текстовые эмбеддинги только для тех изображений, у которых есть описание
        valid_description_indices = [i for i, emb in enumerate(text_description_embeddings) if emb is not None]
        valid_description_embeddings = [emb for emb in text_description_embeddings if emb is not None]
        if valid_description_embeddings:
            valid_description_embeddings = np.array(valid_description_embeddings)
            if len(valid_description_embeddings.shape) == 1:
                valid_description_embeddings = valid_description_embeddings.reshape(-1, query_text_embedding.shape[1])
            valid_distances_descriptions = cdist(query_text_embedding, valid_description_embeddings,
                                                 metric='cosine').flatten()

            # Заполняем расстояния для тех изображений, у которых есть описание
            for idx, valid_idx in enumerate(valid_description_indices):
                distances_descriptions[valid_idx] = valid_distances_descriptions[idx]
        else:
            logger.warning("Все текстовые описания равны None (пустые).")
    else:
        logger.warning("Не было найдено опциональных текстовых описаний.")

    tokenized_corpus = [word_tokenize(text.lower()) for text in ocr_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = word_tokenize(translated_query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to a range of 0-1
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    # Комбинируем расстояния (по изображениям, текстам и знаменитостям)
    combined_distances = (distances_one_peace + distances_ocr + distances_descriptions + bm25_scores) / 4
    indices = np.argsort(combined_distances)[:10]

    # Находим лучшее изображение
    best_image_index = indices[0]
    best_image_name = image_names[best_image_index]

    # Логирование только для лучшего изображения
    logger.info(f"Лучшее изображение: {best_image_name}")
    logger.info(f"  Балл похожести по ONE-PEACE: {1 - distances_one_peace[best_image_index]}")
    logger.info(f"  Балл похожести по тексту OCR: {1 - distances_ocr[best_image_index]}")
    logger.info(f"  Балл похожести по текстовому описанию: {1 - distances_descriptions[best_image_index]}")
    logger.info(f"  Балл BM25: {1 - bm25_scores[best_image_index]}")

    # Формируем результаты поиска
    results = [{"image_name": image_names[i], "similarity": 1 - combined_distances[i]} for i in indices]
    logger.info(f"Поиск по запросу '{query.query}' завершен. Найдено {len(results)} результатов.")
    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
