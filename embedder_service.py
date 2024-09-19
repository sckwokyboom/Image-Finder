import sys

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from sentence_transformers import SentenceTransformer
import torch
import os
import logging
import pytesseract
from typing import Optional

app = FastAPI()

ONE_PEACE_GIT_REPO_PATH = 'ONE-PEACE/'
ONE_PEACE_MODEL_PATH = '/home/meno/models/one-peace.pt'

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


def load_one_peace_model():
    """Загрузка моделей ONE-PEACE и SBERT."""
    if not os.path.isdir(ONE_PEACE_GIT_REPO_PATH):
        raise FileNotFoundError(f'The directory "{ONE_PEACE_GIT_REPO_PATH}" does not exist')
    if not os.path.isfile(ONE_PEACE_MODEL_PATH):
        raise FileNotFoundError(f'The model file "{ONE_PEACE_MODEL_PATH}" does not exist')
    one_peace_dir = os.path.normpath(ONE_PEACE_GIT_REPO_PATH)
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
    logger.info("ONE-PEACE успешно загружен.")
    return model


def load_sbert_model():
    logger.info("Загрузка модели SBERT...")
    model_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("SBERT успешно загружен.")
    return model_sbert


def setup_tesseract_model():
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


@app.on_event("startup")
async def startup_event():
    """Инициализация моделей при старте сервиса."""
    global model_op, model_sbert
    model_op = load_one_peace_model()
    model_sbert = load_sbert_model()
    setup_tesseract_model()


@app.post("/vectorize-image/")
async def vectorize_image(file: UploadFile = File(...), description: Optional[str] = Form(None)):
    """Векторизация изображения с помощью ONE-PEACE."""
    from PIL import Image
    from torchvision import transforms

    image = Image.open(image.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    with torch.no_grad():
        embedding = model_op.extract_image_features(image_tensor).cpu().numpy()

    return {"embedding": embedding.tolist()}


@app.post("/vectorize-text/")
async def vectorize_text(text: str):
    """Векторизация текста с помощью SBERT."""
    text_embedding = model_sbert.encode(text)
    return {"embedding": text_embedding.tolist()}
