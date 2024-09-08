from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
import logging
import os
import sys
import torch
import sqlite3
import faiss
import numpy as np

one_peace_demo_logger = logging.getLogger(__name__)


def vectorize_image(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.extract_image_features(image)
    return embedding.cpu().numpy()


def initialize_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_name TEXT PRIMARY KEY,
            faiss_index INTEGER,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()


def save_embeddings_to_sqlite(db_path, faiss_index, image_name, embedding):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO image_embeddings (faiss_index, image_name, embedding)
        VALUES (?, ?, ?)
    ''', (faiss_index, image_name, embedding.tobytes()))
    conn.commit()
    conn.close()


def save_embeddings_to_faiss(faiss_index, embeddings, faiss_path):
    faiss_index.add(np.array(embeddings))
    faiss.write_index(faiss_index, faiss_path)


def setup_logging(log_path):
    logging.basicConfig(level=logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(stdout_handler)

    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(file_handler)


def main():
    parser = ArgumentParser()
    parser.add_argument('--image-dir', dest='image_dir', type=str, required=True,
                        help='Directory containing images to vectorize.')
    parser.add_argument('--db-path', dest='db_path', type=str, required=True,
                        help='Path to SQLite database.')
    parser.add_argument('--faiss-path', dest='faiss_path', type=str, required=True,
                        help='Path to save FAISS index file.')
    parser.add_argument('--model-dir', dest='model_dir', type=str, required=True, help='Path to ONE-PEACE repository.')
    parser.add_argument('--model-name', dest='model_name', type=str, required=True,
                        help='Path to pre-trained model weights.')
    parser.add_argument('--device', dest='torch_device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for inference.')
    parser.add_argument('--log-path', dest='log_path', type=str, default=None,
                        help='Path to log file.')

    args = parser.parse_args()

    setup_logging(args.log_path)

    device = torch.device('cuda' if torch.cuda.is_available() and args.torch_device != 'cpu' else 'cpu')

    one_peace_dir = os.path.normpath(args.model_dir)
    if not os.path.isdir(one_peace_dir):
        err_msg = f'The dir "{one_peace_dir}" does not exist'
        one_peace_demo_logger.error(err_msg)
        raise ValueError(err_msg)

    model_name = os.path.normpath(args.model_name)
    if not os.path.isfile(model_name):
        err_msg = f'The file "{model_name}" does not exist'
        one_peace_demo_logger.error(err_msg)
        raise ValueError(err_msg)

    sys.path.append(os.path.join(one_peace_dir))
    from one_peace.models import from_pretrained
    one_peace_demo_logger.info('ONE-PEACE model is being loaded.')

    current_workdir = os.getcwd()
    one_peace_demo_logger.info(f'Current workdir: {current_workdir}')
    os.chdir(one_peace_dir)
    one_peace_demo_logger.info(f'New workdir: {os.getcwd()}')

    model = from_pretrained(args.model_name, device=device, dtype=args.torch_device)

    os.chdir(current_workdir)
    one_peace_demo_logger.info(f'Restored workdir: {os.getcwd()}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    initialize_database(args.db_path)

    embeddings = []
    faiss_index = faiss.IndexFlatL2(1536)
    for image_name in os.listdir(args.image_dir):
        image_path = os.path.join(args.image_dir, image_name)
        if os.path.isfile(image_path):
            embedding = vectorize_image(image_path, model, transform, device)
            embeddings.append(embedding.flatten())

            faiss_idx = faiss_index.ntotal
            faiss_index.add(embedding)

            save_embeddings_to_sqlite(args.db_path, faiss_idx, image_name, embedding)

    save_embeddings_to_faiss(faiss_index, embeddings, faiss_path=args.faiss_path)
    one_peace_demo_logger.info('Vectorization process has finished and data saved to SQLite and FAISS.')


if __name__ == '__main__':
    main()
