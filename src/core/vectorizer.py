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
import easyocr
from transformers import AutoTokenizer, AutoModel

one_peace_demo_logger = logging.getLogger(__name__)


def vectorize_image(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.extract_image_features(image)
    return embedding.cpu().numpy()


def extract_text_embedding(image_path, tokenizer, xlm_roberta_model, device):
    model_storage_dir = "/userspace/kmy/image_rag/models/easy_ocr"
    reader = easyocr.Reader(['en', 'ru'], model_storage_directory=model_storage_dir,
                            user_network_directory=model_storage_dir)

    image = Image.open(image_path).convert("RGB")
    result = reader.readtext(np.array(image), detail=0)
    text = " ".join([item for item in result])

    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embedding = xlm_roberta_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        return text, text_embedding
    else:
        return "", None


def initialize_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_name TEXT PRIMARY KEY,
            embedding BLOB,
            recognized_text TEXT,
            text_embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()


def save_embeddings_to_sqlite(db_path, image_name, image_embedding, text, text_embedding):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO image_embeddings (image_name, embedding, recognized_text, text_embedding)
        VALUES (?, ?, ?, ?)
    ''', (
        image_name, image_embedding.tobytes(), text, text_embedding.tobytes() if text_embedding is not None else None))
    conn.commit()
    conn.close()


def add_to_annoy_index(annoy_index, embeddings, image_index):
    annoy_index.add_item(image_index, embeddings)


def save_annoy_index(annoy_index, index_path, n_trees=10):
    annoy_index.build(n_trees)
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
    parser.add_argument('--image-annoy-index-path', dest='image_annoy_index_path', type=str, required=True,
                        help='Path to image Annoy index file.')
    parser.add_argument('--text-annoy-index-path', dest='text_annoy_index_path', type=str, required=True,
                        help='Path to text Annoy index file.')

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
    os.chdir(one_peace_dir)
    model = from_pretrained(args.model_name, device=device, dtype=args.torch_device)
    os.chdir(current_workdir)

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    xlm_roberta_model = AutoModel.from_pretrained('xlm-roberta-base').to(device)

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

        image_annoy_index = AnnoyIndex(embedding_dim, 'angular')
        text_annoy_index = AnnoyIndex(768, 'angular')
    else:
        one_peace_demo_logger.error("No images found in the directory.")
        sys.exit(1)

    for idx, image_name in enumerate(image_dir_files):
        image_path = os.path.join(args.image_dir, image_name)
        if os.path.isfile(image_path):
            image_embedding = vectorize_image(image_path, model, transform, device)
            text, text_embedding = extract_text_embedding(image_path, tokenizer, xlm_roberta_model, device)

            save_embeddings_to_sqlite(args.db_path, image_name, image_embedding, text, text_embedding)
            add_to_annoy_index(image_annoy_index, image_embedding[0], idx)

            if text_embedding is not None:
                add_to_annoy_index(text_annoy_index, text_embedding[0], idx)

    save_annoy_index(image_annoy_index, args.image_annoy_index_path)
    save_annoy_index(text_annoy_index, args.text_annoy_index_path)

    one_peace_demo_logger.info(f'Image Annoy index saved to {args.image_annoy_index_path}')
    one_peace_demo_logger.info(f'Text Annoy index saved to {args.text_annoy_index_path}')
    one_peace_demo_logger.info('Vectorization and indexing process has finished.')


if __name__ == '__main__':
    main()
