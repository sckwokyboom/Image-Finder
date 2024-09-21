import sys
import os

import pytesseract
import torch
import sqlite3
import numpy as np
import logging
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from scipy.spatial.distance import cdist
from deep_translator import GoogleTranslator
from typing import Optional
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from annoy import AnnoyIndex
from hashlib import md5

IMAGE_DIR = os.path.abspath("/home/meno/image_rag/Image-RAG/resources/val2017")
DB_PATH = os.path.abspath("/home/meno/image_rag/Image-RAG/resources/images_metadata.db")
ONE_PEACE_GIT_REPO_DIR_PATH = 'ONE-PEACE/'
ONE_PEACE_MODEL_PATH = '/home/meno/models/one-peace.pt'
ONE_PEACE_EMBEDDING_SIZE = 768
TEXT_EMBEDDING_SIZE = 384

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


def translate_to_english(text):
    translator = GoogleTranslator(source='ru', target='en')
    translated_text = translator.translate(text)
    return translated_text


def compute_hash(embedding: np.ndarray) -> str:
    """Вычисление хэша для эмбеддинга."""
    return md5(embedding.tobytes()).hexdigest()


def save_hash_to_db(image_path, emb_hash, column_name: str):
    conn = sqlite3.connect(DB_PATH)
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


def add_to_annoy_index(index, embedding, image_id):
    """Добавить эмбеддинг в ANNOY-индекс."""
    index.add_item(image_id, embedding)


def build_annoy_index(embeddings, embedding_size, index_path):
    """Создание и сохранение ANNOY-индекса."""
    index = AnnoyIndex(embedding_size, 'angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10)  # Количество деревьев для точности поиска
    index.save(index_path)
    logger.info(f"ANNOY индекс сохранен в {index_path}")


def load_annoy_index(dimension, file_path):
    """Загрузка ANNOY-индекса или создание нового, если файла нет."""
    index = AnnoyIndex(dimension, 'angular')
    if os.path.exists(file_path):
        logger.info(f"Загружаем ANNOY-индекс из файла {file_path}.")
        index.load(file_path)
    else:
        logger.warning(f"Файл {file_path} не найден. Создаём новый ANNOY-индекс.")
    return index


def create_all_annoy_indexes(db_path):
    """Создание ANNOY-индексов для всех метрик."""
    image_names, image_embeddings, ocr_texts, text_descriptions, text_description_embeddings \
        = get_image_embeddings(db_path)

    # Создание ANNOY-индекса для эмбеддингов One-PEACE
    build_annoy_index(image_embeddings, ONE_PEACE_EMBEDDING_SIZE, 'one_peace.ann')

    # Создание ANNOY-индекса для эмбеддингов OCR текстов
    ocr_embeddings = model_sbert.encode(ocr_texts)
    build_annoy_index(ocr_embeddings, TEXT_EMBEDDING_SIZE, 'ocr.ann')

    # Создание ANNOY-индекса для текстовых описаний
    valid_description_embeddings = [emb for emb in text_description_embeddings if emb is not None]
    if valid_description_embeddings:
        build_annoy_index(valid_description_embeddings, TEXT_EMBEDDING_SIZE, 'descriptions.ann')


def setup_models():
    """Загрузка модели ONE-PEACE с проверкой путей."""
    if not os.path.isdir(ONE_PEACE_GIT_REPO_DIR_PATH):
        raise FileNotFoundError(f'The directory "{ONE_PEACE_GIT_REPO_DIR_PATH}" does not exist')
    if not os.path.isfile(ONE_PEACE_MODEL_PATH):
        raise FileNotFoundError(f'The model file "{ONE_PEACE_MODEL_PATH}" does not exist')

    # nltk.data.path.append('/home/meno/models/nltk_data/tokenizers/punkt')
    nltk.download('punkt_tab')

    one_peace_dir = os.path.normpath(ONE_PEACE_GIT_REPO_DIR_PATH)
    if not os.path.isdir(one_peace_dir):
        err_msg = f'The dir "{one_peace_dir}" does not exist'
        logger.error(err_msg)
        raise ValueError(err_msg)

    model_name = os.path.normpath(ONE_PEACE_MODEL_PATH)
    if not os.path.isfile(model_name):
        err_msg = f'The file "{model_name}" does not exist'
        logger.error(err_msg)
        raise ValueError(err_msg)
    sys.path.append(one_peace_dir)
    from one_peace.models import from_pretrained

    logger.info("Загрузка модели ONE-PEACE...")
    current_workdir = os.getcwd()
    logger.info(f'Текущая рабочая директория: {current_workdir}')

    os.chdir(one_peace_dir)
    logger.info(f'Новая рабочая директория: {os.getcwd()}')
    model = from_pretrained(model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info("ONE-PEACE был успешно загружен")
    logger.info("Загрузка модели SBERT...")
    model_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("SBERT был успешно загружен")
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
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
            op_embedding_hash TEXT,
            recognized_text TEXT,
            recognized_text_embedding BLOB,
            recognized_text_embedding_hash TEXT,
            text_description TEXT,
            text_description_embedding BLOB,
            text_description_embedding_hash TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Инициализация базы данных завершена")


def save_embedding(db_path,
                   image_name: str,
                   op_embedding,
                   op_embedding_hash,
                   recognized_text,
                   recognized_text_embedding,
                   recognized_text_embedding_hash,
                   textual_description,
                   textual_description_embedding,
                   textual_description_embedding_hash):
    """Сохранение эмбеддингов и текста OCR в базу данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO image_embeddings (image_name, op_embedding, op_embedding_hash, recognized_text, recognized_text_embedding, recognized_text_embedding_hash, text_description, text_description_embedding, text_description_embedding_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        image_name,
        op_embedding.tobytes(),
        op_embedding_hash,
        recognized_text,
        recognized_text_embedding.tobytes() if recognized_text_embedding is not None else None,
        recognized_text_embedding_hash,
        textual_description,
        textual_description_embedding.tobytes() if textual_description_embedding is not None else None,
        textual_description_embedding_hash

    ))
    conn.commit()
    conn.close()


def get_image_embeddings(db_path):
    """Получение всех эмбеддингов изображений и текстов OCR из базы данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT image_name, op_embedding, recognized_text, text_description, text_description_embedding FROM image_embeddings')
    results = cursor.fetchall()
    conn.close()

    image_names = [row[0] for row in results]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in results]
    ocr_texts = [row[2] for row in results]
    textual_descriptions = [row[3] for row in results]
    textual_descriptions_embeddings = [np.frombuffer(row[4], dtype=np.float32) if row[4] is not None else None for row
                                       in results]

    return image_names, np.vstack(embeddings), ocr_texts, textual_descriptions, np.vstack(
        textual_descriptions_embeddings)


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
    global model_op, model_sbert, transform_op, one_peace_index, ocr_index, description_index
    initialize_database(DB_PATH)
    model_op, model_sbert = setup_models()
    transform_op = create_transforms()
    one_peace_index = load_annoy_index(ONE_PEACE_EMBEDDING_SIZE, 'one_peace.ann')
    ocr_index = load_annoy_index(TEXT_EMBEDDING_SIZE, 'ocr.ann')
    description_index = load_annoy_index(TEXT_EMBEDDING_SIZE, 'descriptions.ann')

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


# def get_image_embeddings(db_path):
#     """Получение всех эмбеддингов изображений, текстов OCR, описаний и имен знаменитостей из базы данных."""
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute(
#         'SELECT image_name, op_embedding, recognized_text, text_description_embedding FROM image_embeddings')
#     results = cursor.fetchall()
#     conn.close()
#
#     image_names = [row[0] for row in results]
#     embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in results]
#     # TODO: очень плохо с индексами тут работать, явно неправильно
#     ocr_texts = [row[2] for row in results]
#     text_description_embeddings = [np.frombuffer(row[3], dtype=np.float32) if row[3] is not None else None for row in
#                                    results]
#
#     return image_names, np.vstack(embeddings), ocr_texts, text_description_embeddings


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), description: Optional[str] = Form(None)):
    """Загрузка изображения и сохранение его эмбеддингов."""
    logger.info("Изображение получено.")
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=400, detail="Невозможно обработать изображение")

    op_embedding = vectorize_image(model_op, transform_op, image,
                                   torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    op_embedding_hash = compute_hash(op_embedding)
    # Использование pytesseract для распознавания текста
    ocr_result = pytesseract.image_to_string(image, lang='eng+rus')
    ocr_embedding = None
    ocr_embedding_hash = None
    ocr_text = ocr_result.strip()
    if ocr_text:
        logging.info(f"Распознанный текст: {ocr_text}")
        ocr_embedding = model_sbert.encode(ocr_text)
        ocr_embedding_hash = compute_hash(ocr_embedding)
    else:
        logging.warning("Не удалось распознать текст на изображении")

    textual_description_embedding = None
    textual_description_embedding_hash = None
    if description:
        logging.info(f"Получено описание изображения: {description}")
        textual_description_embedding = model_sbert.encode(description)
        textual_description_embedding_hash = compute_hash(textual_description_embedding)
    else:
        logging.warning("Описание изображения отсутствует")

    next_image_number = get_next_image_number(IMAGE_DIR)
    new_image_name = f"{next_image_number:05d}.jpg"
    image_path = os.path.join(IMAGE_DIR, new_image_name)

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    image.save(image_path)
    save_embedding(DB_PATH, new_image_name, op_embedding, op_embedding_hash, ocr_text, ocr_embedding,
                   ocr_embedding_hash, description, textual_description_embedding, textual_description_embedding_hash)

    add_to_annoy_index(one_peace_index, op_embedding, next_image_number)
    if ocr_embedding is not None:
        add_to_annoy_index(ocr_index, ocr_embedding, next_image_number)

    if textual_description_embedding is not None:
        add_to_annoy_index(description_index, textual_description_embedding, next_image_number)

    # Построение индексов после добавления новых эмбеддингов (делаем это только после загрузки нескольких изображений для оптимизации)
    if next_image_number % 1 == 0:  # Например, строить индекс каждые 10 изображений
        logger.info("Строим ANNOY-индексы...")
        one_peace_index.build(10)  # Число деревьев для поиска, может варьироваться
        ocr_index.build(10)
        description_index.build(10)

        # Сохраняем обновленные индексы на диск
        one_peace_index.save("one_peace_index.ann")
        ocr_index.save("ocr_index.ann")
        description_index.save("description_index.ann")
        logger.info(f"Индексы построены и записаны в файлы.")

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

    one_peace_neighbors = one_peace_index.get_nns_by_vector(text_features, 10, include_distances=True)
    one_peace_image_hashes = [compute_hash(one_peace_index.get_item_vector(i)) for i in
                              one_peace_neighbors[0]]
    one_peace_distances = one_peace_neighbors[1]
    one_peace_image_names = get_image_names_by_hashes(DB_PATH, one_peace_image_hashes, "one-peace")

    ocr_neighbors = ocr_index.get_nns_by_vector(query_text_embedding, 10, include_distances=True)
    ocr_image_hashes = [compute_hash(ocr_index.get_item_vector(i)) for i in ocr_neighbors[0]]
    ocr_distances = ocr_neighbors[1]
    ocr_image_names = get_image_names_by_hashes(DB_PATH, ocr_image_hashes, "ocr")

    logger.info("Поиск по эмбеддингам текстовых описаний через ANNOY")
    description_neighbors = description_index.get_nns_by_vector(query_text_embedding, 10, include_distances=True)
    description_distances = description_neighbors[1]
    description_image_hashes = [compute_hash(description_index.get_item_vector(i)) for i in
                                description_neighbors[0]]
    description_image_names = get_image_names_by_hashes(DB_PATH, description_image_hashes, "description")
    combined_image_names = set(one_peace_image_names + ocr_image_names + description_image_names)
    if not combined_image_names:
        return JSONResponse({"status": "Изображения не найдены."}, status_code=404)

    combined_results = {}

    # Для каждого изображения из one-peace
    for i, image_name in enumerate(one_peace_image_names):
        if image_name not in combined_results:
            combined_results[image_name] = {'one_peace_distance': one_peace_distances[i], 'ocr_distance': None,
                                            'description_distance': None}
        else:
            combined_results[image_name]['one_peace_distance'] = one_peace_distances[i]

    # Для каждого изображения из ocr
    for i, image_name in enumerate(ocr_image_names):
        if image_name not in combined_results:
            combined_results[image_name] = {'one_peace_distance': None, 'ocr_distance': ocr_distances[i],
                                            'description_distance': None}
        else:
            combined_results[image_name]['ocr_distance'] = ocr_distances[i]

    # Для каждого изображения из описаний
    for i, image_name in enumerate(description_image_names):
        if image_name not in combined_results:
            combined_results[image_name] = {'one_peace_distance': None, 'ocr_distance': None,
                                            'description_distance': description_distances[i]}
        else:
            combined_results[image_name]['description_distance'] = description_distances[i]

    results = []
    for image_name, distances in combined_results.items():
        # Считаем только те метрики, которые существуют
        valid_distances = [dist for dist in [distances['one_peace_distance'], distances['ocr_distance'],
                                             distances['description_distance']] if dist is not None]
        if valid_distances:
            avg_distance = sum(valid_distances) / len(valid_distances)
            results.append({'image_name': image_name, 'avg_distance': avg_distance})

    # Сортируем по средней дистанции
    sorted_results = sorted(results, key=lambda x: x['avg_distance'])[:10]

    if not sorted_results:
        return JSONResponse({"status": "Изображения не найдены."}, status_code=404)

    logger.info(f"Поиск по запросу '{query.query}' завершен. Найдено {len(results)} результатов.")
    return JSONResponse(sorted_results)


def get_image_names_by_hashes(db_path: str, embedding_hashes: list[str], index_type: str) -> list:
    """
    Находит изображения по хэшу эмбеддинга, извлеченного из ANNOY индекса.

    :param db_path: Путь к базе данных
    :param annoy_index: ANNOY индекс, используемый для поиска ближайших эмбеддингов
    :param query_embedding: Эмбеддинг, по которому производится поиск
    :param index_type: Тип индекса (one-peace, ocr, description), чтобы искать по соответствующему столбцу
    :return: Список имен изображений, которые имеют совпадающий хэш эмбеддинга
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    image_names = []

    # В зависимости от типа индекса используем разные столбцы для поиска хэшей
    if index_type == 'one-peace':
        hash_column = 'one_peace_hash'
    elif index_type == 'ocr':
        hash_column = 'ocr_hash'
    elif index_type == 'description':
        hash_column = 'description_hash'
    else:
        raise ValueError(f"Неизвестный тип индекса: {index_type}")

    # Для каждого найденного эмбеддинга ищем изображения с этим хэшем
    for vector_hash in embedding_hashes:
        cursor.execute(f"SELECT image_name FROM images WHERE {hash_column} = ?", (vector_hash,))
        result = cursor.fetchone()
        if result:
            image_names.append(result[0])

    conn.close()

    if not image_names:
        raise ValueError(f"Изображения с хэшами эмбеддингов не найдены.")

    return image_names


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
