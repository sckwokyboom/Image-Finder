from argparse import ArgumentParser
import logging
import os
import sys
import json
import torch

one_peace_demo_logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('--embeddings-file', dest='embeddings_file', type=str, required=True,
                        help='File containing precomputed image embeddings.')
    parser.add_argument('--model-dir', dest='model_dir', type=str, required=True, help='Path to ONE-PEACE repository.')
    parser.add_argument('--model-name', dest='model_name', type=str, required=True,
                        help='Path to pre-trained model weights.')
    parser.add_argument('--device', dest='torch_device', type=str, default='cuda', choices=['cuda', 'cpu', 'gpu'],
                        help='Device to use for inference.')

    args = parser.parse_args()

    if args.torch_device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        if args.torch_device not in {'cpu', 'cuda', 'gpu'}:
            err_msg = f'The device "{args.torch_device}" is unknown!'
            one_peace_demo_logger.error(err_msg)
            raise ValueError(err_msg)
        if (not torch.cuda.is_available()) and (args.torch_device in {'cuda', 'gpu'}):
            err_msg = f'The device "{args.torch_device}" is not available!'
            one_peace_demo_logger.error(err_msg)
            raise ValueError(err_msg)
        device = 'cpu' if (args.torch_device == 'cpu') else 'cuda'
        one_peace_demo_logger.info(f'{device.upper()} is used.')

        one_peace_dir = os.path.normpath(args.model_dir)
        if not os.path.isdir(one_peace_dir):
            err_msg = f'The directory "{one_peace_dir}" does not exist!'
            one_peace_demo_logger.error(err_msg)
            raise ValueError(err_msg)

        model_name = os.path.normpath(args.model_name)
        if not os.path.isfile(model_name):
            err_msg = f'The file "{model_name}" does not exist!'
            one_peace_demo_logger.error(err_msg)
            raise ValueError(err_msg)

        sys.path.append(os.path.join(one_peace_dir))
        from one_peace.models import from_pretrained
        one_peace_demo_logger.info('ONE-PEACE is attached.')

        current_workdir = os.getcwd()
        one_peace_demo_logger.info(f'Current working directory: {current_workdir}')
        os.chdir(one_peace_dir)
    one_peace_demo_logger.info(f'New working directory: {os.getcwd()}')
    model = from_pretrained(model_name, device=device, dtype=args.torch_device)
    one_peace_demo_logger.info('Model is loaded.')
    os.chdir(current_workdir)
    one_peace_demo_logger.info(f'Restored working directory: {os.getcwd()}')

    with open(args.embeddings_file, 'r') as f:
        image_embeddings = json.load(f)

    query_text = ['A girl plays badminton in the heat.']
    text_tokens = model.process_text(query_text)
    with torch.no_grad():
        text_features = model.extract_text_features(text_tokens)

    best_match = None
    best_similarity = -float('inf')
    for image_name, image_embedding in image_embeddings.items():
        image_embedding_tensor = torch.tensor(image_embedding).to(device)
        similarity = image_embedding_tensor @ text_features.T
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = image_name

    one_peace_demo_logger.info(f'Best match: {best_match} with similarity {best_similarity.item()}')


if __name__ == '__main__':
    one_peace_demo_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('one_peace_search_demo.log')
    file_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(file_handler)
    main()
