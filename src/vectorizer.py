from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
import logging
import os
import sys
import json
import torch

one_peace_demo_logger = logging.getLogger(__name__)


def vectorize_image(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.extract_image_features(image)
    return embedding


def main():
    parser = ArgumentParser()
    parser.add_argument('--image-dir', dest='image_dir', type=str, required=True,
                        help='Directory containing images to vectorize.')
    parser.add_argument('--output-file', dest='output_file', type=str, required=True,
                        help='File to save the image embeddings.')
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

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_embeddings = {}
    for image_name in os.listdir(args.image_dir):
        image_path = os.path.join(args.image_dir, image_name)
        if os.path.isfile(image_path):
            embedding = vectorize_image(image_path, model, transform, device)
            image_embeddings[image_name] = embedding.cpu().numpy().tolist()

    one_peace_demo_logger.info(f'Vectorize process has finished.')

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w') as f:
        json.dump(image_embeddings, f)


if __name__ == '__main__':
    one_peace_demo_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('one_peace_embeddings_vectorizing.log')
    file_handler.setFormatter(formatter)
    one_peace_demo_logger.addHandler(file_handler)
    main()