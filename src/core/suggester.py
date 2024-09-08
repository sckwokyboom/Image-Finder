from argparse import ArgumentParser
import logging
import os
import sys
import torch
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist

one_peace_demo_logger = logging.getLogger(__name__)

def get_image_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT image_name, embedding FROM image_embeddings')
    results = cursor.fetchall()
    conn.close()

    image_names = []
    embeddings = []
    for row in results:
        image_names.append(row[0])
        embeddings.append(np.frombuffer(row[1], dtype=np.float32))

    return image_names, np.vstack(embeddings)

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
    parser.add_argument('--db-path', dest='db_path', type=str, required=True,
                        help='Path to SQLite database with image embeddings.')
    parser.add_argument('--model-dir', dest='model_dir', type=str, required=True,
                        help='Path to ONE-PEACE repository.')
    parser.add_argument('--model-name', dest='model_name', type=str, required=True,
                        help='Path to pre-trained model weights.')
    parser.add_argument('--device', dest='torch_device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for inference.')
    parser.add_argument('--query', dest='query', type=str, required=True,
                        help='Text query for image search.')
    parser.add_argument('--top-k', dest='top_k', type=int, default=5,
                        help='Number of top results to return.')
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
    one_peace_demo_logger.info('Loading ONE-PEACE model.')

    current_workdir = os.getcwd()
    one_peace_demo_logger.info(f'Current workdir: {current_workdir}')
    os.chdir(one_peace_dir)
    one_peace_demo_logger.info(f'New workdir: {os.getcwd()}')

    model = from_pretrained(args.model_name, device=device, dtype=args.torch_device)

    query_text = [args.query]
    text_tokens = model.process_text(query_text)
    with torch.no_grad():
        text_features = model.extract_text_features(text_tokens).cpu().numpy()

    os.chdir(current_workdir)
    one_peace_demo_logger.info(f'Restored workdir: {os.getcwd()}')

    image_names, image_embeddings = get_image_embeddings(args.db_path)

    distances = cdist(text_features, image_embeddings, metric='cosine').flatten()
    indices = np.argsort(distances)[:args.top_k]

    one_peace_demo_logger.info(f'Top {args.top_k} results for query "{args.query}":')
    for i, idx in enumerate(indices):
        image_name = image_names[idx]
        similarity = 1 - distances[idx]
        one_peace_demo_logger.info(f'{i + 1}. Image: {image_name}, Similarity: {similarity:.4f}')

if __name__ == '__main__':
    main()
