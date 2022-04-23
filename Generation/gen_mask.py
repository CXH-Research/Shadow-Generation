import utils as utils
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os
from tqdm import trange


def save_image(tensor, mask, dir, name):
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    if mask:
        tensor = Image.fromarray(tensor).convert("L")
    else:
        tensor = Image.fromarray(tensor).convert("RGB")
    tensor.save(os.path.join(dir, name))


parser = argparse.ArgumentParser(
    description='This is the composition method to generate shadow')
parser.add_argument('--min_val', type=int, default=0.7,
                    help='Render per-pixel intensity variation mask within [min_val, 1.]')
parser.add_argument('--height', type=int, default=256,
                    help='Image height')
parser.add_argument('--width', type=int, default=256,
                    help='Image width')
parser.add_argument('--num_mask', type=int, default=10,
                    help='Number of mask you want to generate')
args = parser.parse_args()

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

TONE_SIGMA = 0.1
SS_SIGMA = 0.5

size = (args.height, args.width)

for i in trange(args.num_mask):
    min_val = random.uniform(args.min_val, 1)
    intensity_mask = utils.get_brightness_mask(size=size, min_val=0.7)
    save_image(intensity_mask, True, 'mask', str(i) + '.png')
