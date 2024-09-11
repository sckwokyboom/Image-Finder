from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
import logging
import os
import sys
import torch
import sqlite3
import numpy as np
from annoy import AnnoyIndex

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
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

def save_embeddings_to_sqlite(db_path, image_name, embedding):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO image_embeddings (image_name, embedding)
        VALUES (?, ?)
    ''', (image_name, embedding.tobytes()))
    conn.commit()
    conn.close()

def update_annoy_index(annoy_index, image_embeddings, image_index, n_trees=10):
    annoy_index.add_item(image_index, image_embeddings)
    annoy_index.build(n_trees)

def save_annoy_index(annoy_index, index_path):
    annoy_index.save(index_path)

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
    parser.add_argument('--model-dir', dest='model_dir', type=str, required=True, help='Path to ONE-PEACE repository.')
    parser.add_argument('--model-name', dest='model_name', type=str, required=True,
                        help='Path to pre-trained model weights.')
    parser.add_argument('--device', dest='torch_device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for inference.')
    parser.add_argument('--log-path', dest='log_path', type=str, default=None,
                        help='Path to log file.')
    parser.add_argument('--annoy-index-path', dest='annoy_index_path', type=str, required=True,
                        help='Path to Annoy index file.')

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

    image_dir_files = os.listdir(args.image_dir)
    if image_dir_files:
        first_image_path = os.path.join(args.image_dir, image_dir_files[0])
        first_embedding = vectorize_image(first_image_path, model, transform, device)
        embedding_dim = first_embedding.shape[1]

        annoy_index = AnnoyIndex(embedding_dim, 'angular')  # Угловая метрика
    else:
        one_peace_demo_logger.error("No images found in the directory.")
        sys.exit(1)

    for idx, image_name in enumerate(image_dir_files):
        image_path = os.path.join(args.image_dir, image_name)
        if os.path.isfile(image_path):
            embedding = vectorize_image(image_path, model, transform, device)
            save_embeddings_to_sqlite(args.db_path, image_name, embedding)
            update_annoy_index(annoy_index, embedding[0], idx)

    save_annoy_index(annoy_index, args.annoy_index_path)
    one_peace_demo_logger.info(f'Annoy index saved to {args.annoy_index_path}')
    one_peace_demo_logger.info('Vectorization and indexing process has finished.')

if __name__ == '__main__':
    main()
