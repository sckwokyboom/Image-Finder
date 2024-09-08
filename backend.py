import sys
import os
import torch
import sqlite3
import numpy as np
import logging
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import transforms
from scipy.spatial.distance import cdist

IMAGE_DIR = "resources/images"
DB_PATH = "resources/images_metadata.db"
MODEL_DIR = 'ONE-PEACE/'
MODEL_NAME = 'C:\\Users\\sckwo\\PycharmProjects\\Image-RAG\\models\\one-peace.pt'

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model(model_dir, model_name):
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
    logger.info(f'Current workdir: {current_workdir}')

    os.chdir(one_peace_dir)
    logger.info(f'New workdir: {os.getcwd()}')
    model = from_pretrained(model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model


def initialize_database(db_path):
    """Создание базы данных, если она не существует."""
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_name TEXT PRIMARY KEY,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()


def save_embedding(db_path, image_name, embedding):
    """Сохранение эмбеддингов изображения в базу данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO image_embeddings (image_name, embedding)
        VALUES (?, ?)
    ''', (image_name, embedding.tobytes()))
    conn.commit()
    conn.close()


def get_image_embeddings(db_path):
    """Получение всех эмбеддингов изображений из базы данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT image_name, embedding FROM image_embeddings')
    results = cursor.fetchall()
    conn.close()

    image_names = [row[0] for row in results]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in results]

    return image_names, np.vstack(embeddings)


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
    global model_op, transform_op
    initialize_database(DB_PATH)
    model_op = setup_model(MODEL_DIR, MODEL_NAME)
    transform_op = create_transforms()
    logger.info("Сервер запущен и готов к работе.")


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """Загрузка изображения и сохранение его эмбеддингов."""
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=400, detail="Невозможно обработать изображение")

    embedding = vectorize_image(model_op, transform_op, image, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    image_path = os.path.join(IMAGE_DIR, file.filename)

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    image.save(image_path)
    save_embedding(DB_PATH, file.filename, embedding)

    logger.info(f"Изображение '{file.filename}' загружено и обработано.")
    return JSONResponse({"status": "Image uploaded and processed."})


class QueryRequest(BaseModel):
    query: str


@app.post("/search/")
async def search_images(query: QueryRequest):
    """Поиск изображений по запросу."""
    text_tokens = model_op.process_text([query.query])
    with torch.no_grad():
        text_features = model_op.extract_text_features(text_tokens).cpu().numpy()

    image_names, image_embeddings = get_image_embeddings(DB_PATH)
    distances = cdist(text_features, image_embeddings, metric='cosine').flatten()
    indices = np.argsort(distances)[:10]

    results = [{"image_name": image_names[i], "similarity": 1 - distances[i]} for i in indices]
    logger.info(f"Поиск по запросу '{query.query}' завершен. Найдено {len(results)} результатов.")
    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
