from fastapi import FastAPI, UploadFile
import sqlite3
import numpy as np
from annoy import AnnoyIndex

app = FastAPI()

DB_PATH = "path/to/images_metadata.db"
ANNOY_INDEX_PATH = "path/to/annoy_index.ann"


@app.post("/save-embedding/")
async def save_embedding(image_name: str, embedding: list):
    """Сохранение эмбеддинга изображения в базу данных и Annoy-индекс."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''INSERT OR REPLACE INTO image_embeddings (image_name, op_embedding) VALUES (?, ?)''',
                   (image_name, np.array(embedding).tobytes()))
    conn.commit()
    conn.close()

    # Обновляем Annoy-индекс
    index = AnnoyIndex(len(embedding), 'angular')
    index.load(ANNOY_INDEX_PATH)
    index.add_item(len(index), embedding)
    index.build(10)  # Параметр может варьироваться
    index.save(ANNOY_INDEX_PATH)

    return {"status": "Embedding saved and index updated"}


@app.post("/search-embedding/")
async def search_embedding(embedding: list):
    """Поиск похожих изображений с помощью Annoy."""
    index = AnnoyIndex(len(embedding), 'angular')
    index.load(ANNOY_INDEX_PATH)
    results = index.get_nns_by_vector(embedding, 10, include_distances=True)

    return {"results": results}
