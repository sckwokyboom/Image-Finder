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
from transformers import AutoTokenizer, AutoModel

one_peace_demo_logger = logging.getLogger(__name__)


def get_image_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT image_name, embedding, text_embedding FROM image_embeddings')
    results = cursor.fetchall()
    conn.close()

    image_names = []
    image_embeddings = []
    text_embeddings = []
    for row in results:
        image_names.append(row[0])
        image_embeddings.append(np.frombuffer(row[1], dtype=np.float32))
        if row[2]:
            text_embeddings.append(np.frombuffer(row[2], dtype=np.float32))
        else:
            text_embeddings.append(None)
    return image_names, np.vstack(image_embeddings), text_embeddings


def load_annoy_index(index_path, embedding_dim):
    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    annoy_index.load(index_path)
    return annoy_index


def search_annoy_index(annoy_index, query_embedding, top_k=5):
    indices, distances = annoy_index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
    return indices, distances


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


def get_xlm_roberta_embedding(query, model, tokenizer, device):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument('--db-path', dest='db_path', type=str, required=True,
                        help='Path to SQLite database with image embeddings.')
    parser.add_argument('--model-dir', dest='model_dir', type=str, required=True,
                        help='Path to ONE-PEACE repository.')
    parser.add_argument('--model-name', dest='model_name', type=str, required=True,
                        help='Path to pre-trained ONE-PEACE model weights.')
    # parser.add_argument('--xlm-roberta-path', dest='xlm_roberta_path', type=str, required=True,
    #                     help='Path to XLM-RoBERTa model (e.g., "xlm-roberta-base").')
    parser.add_argument('--device', dest='torch_device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for inference.')
    parser.add_argument('--query', dest='query', type=str, required=True,
                        help='Text query for image search.')
    parser.add_argument('--top-k', dest='top_k', type=int, default=5,
                        help='Number of top results to return.')
    parser.add_argument('--image-annoy-index-path', dest='image_annoy_index_path', type=str, required=True,
                        help='Path to image Annoy index file.')
    parser.add_argument('--text-annoy-index-path', dest='text_annoy_index_path', type=str, required=True,
                        help='Path to text Annoy index file.')
    parser.add_argument('--log-path', dest='log_path', type=str, default=None,
                        help='Path to log file.')

    args = parser.parse_args()

    setup_logging(args.log_path)

    device = torch.device('cuda' if torch.cuda.is_available() and args.torch_device != 'cpu' else 'cpu')

    # Load ONE-PEACE model
    one_peace_dir = os.path.normpath(args.model_dir)
    sys.path.append(os.path.join(one_peace_dir))
    from one_peace.models import from_pretrained
    one_peace_demo_logger.info('Loading ONE-PEACE model.')

    current_workdir = os.getcwd()
    os.chdir(one_peace_dir)
    model = from_pretrained(args.model_name, device=device, dtype=args.torch_device)

    # Load XLM-RoBERTa model and tokenizer
    one_peace_demo_logger.info('Loading XLM-RoBERTa model.')
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    roberta_model = AutoModel.from_pretrained('xlm-roberta-base').to(device)

    # Get embeddings for query
    query_text = [args.query]

    # ONE-PEACE text embedding
    text_tokens = model.process_text(query_text)
    with torch.no_grad():
        one_peace_text_features = model.extract_text_features(text_tokens).cpu().numpy()

    # XLM-RoBERTa text embedding
    xlm_roberta_features = get_xlm_roberta_embedding(query_text, roberta_model, tokenizer, device)

    # Load Annoy indices
    image_names, image_embeddings, text_embeddings = get_image_embeddings(args.db_path)

    if os.path.exists(args.image_annoy_index_path):
        one_peace_demo_logger.info(f'Loading existing image Annoy index from {args.image_annoy_index_path}.')
        embedding_dim = one_peace_text_features.shape[1]
        image_annoy_index = load_annoy_index(args.image_annoy_index_path, embedding_dim)
        indices, distances = search_annoy_index(image_annoy_index, one_peace_text_features[0], top_k=args.top_k)
        one_peace_demo_logger.info(f'Top {args.top_k} results for ONE-PEACE query "{args.query}":')
        for i, idx in enumerate(indices):
            image_name = image_names[idx]
            similarity = 1 - distances[i]
            one_peace_demo_logger.info(f'{i + 1}. Image: {image_name}, Similarity: {similarity:.4f}')
    else:
        one_peace_demo_logger.error("ONE-PEACE Annoy index does not exist.")

    if os.path.exists(args.text_annoy_index_path):
        one_peace_demo_logger.info(f'Loading existing text Annoy index from {args.text_annoy_index_path}.')
        text_embedding_dim = xlm_roberta_features.shape[1]
        text_annoy_index = load_annoy_index(args.text_annoy_index_path, text_embedding_dim)
        indices, distances = search_annoy_index(text_annoy_index, xlm_roberta_features[0], top_k=args.top_k)
        one_peace_demo_logger.info(f'Top {args.top_k} results for XLM-RoBERTa query "{args.query}":')
        for i, idx in enumerate(indices):
            image_name = image_names[idx]
            similarity = 1 - distances[i]
            one_peace_demo_logger.info(f'{i + 1}. Image: {image_name}, Similarity: {similarity:.4f}')
    else:
        one_peace_demo_logger.error("XLM-RoBERTa Annoy index does not exist.")


if __name__ == '__main__':
    main()
