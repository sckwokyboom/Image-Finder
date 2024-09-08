from argparse import ArgumentParser
import logging
import os
import sys
import torch
import faiss
import sqlite3

one_peace_demo_logger = logging.getLogger(__name__)


def get_image_name_by_faiss_index(db_path, faiss_index):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT image_name FROM image_embeddings WHERE faiss_index=?', (faiss_index,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

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
    parser.add_argument('--faiss-path', dest='faiss_path', type=str, required=True,
                        help='Path to FAISS index file.')
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

    if not os.path.isfile(args.faiss_path):
        err_msg = f'The FAISS index file "{args.faiss_path}" does not exist!'
        one_peace_demo_logger.error(err_msg)
        raise ValueError(err_msg)

    faiss_index = faiss.read_index(args.faiss_path)

    distances, indices = faiss_index.search(text_features, args.top_k)

    one_peace_demo_logger.info(f'Top {args.top_k} results for query "{args.query}":')
    for i, idx in enumerate(indices[0]):
        image_name = get_image_name_by_faiss_index(args.db_path, idx)
        if image_name:
            distance = distances[0][i]
            similarity = 1 / (1 + distance)
            one_peace_demo_logger.info(f'{i + 1}. Image: {image_name}, Similarity: {similarity:.4f}')
        else:
            one_peace_demo_logger.warning(f'No image found for FAISS index {idx}')


if __name__ == '__main__':
    main()
