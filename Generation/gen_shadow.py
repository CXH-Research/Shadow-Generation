import utils as utils
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser(
    description='This is the composition method to generate shadow')
parser.add_argument('--min_val', type=int, default=0.7,
                    help='Render per-pixel intensity variation mask within [min_val, 1.]')
parser.add_argument('--height', type=int, default=256,
                    help='Image height')
parser.add_argument('--width', type=int, default=256,
                    help='Image width')
args = parser.parse_args()

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
TONE_SIGMA = 0.1
SS_SIGMA = 0.5

size = (args.height, args.width)

inputs = os.listdir('input')

for inp in inputs:
    img_format = inp.split('.')[-1]
    if img_format == 'png':
        bg = utils.read_float(os.path.join('input', inp),
                              channel=3, itype='png', is_linear=False)
    else:
        bg = utils.read_float(os.path.join('input', inp),
                              channel=3, itype='jpg', is_linear=False)
    # create mask
    min_val = random.uniform(args.min_val, 1)
    intensity_mask = utils.get_brightness_mask(size=size, min_val=0.7)
    tf.gfile.FastGFile(os.path.join('mask', inp), intensity_mask)
    bg = cv2.resize(bg.numpy(), size)
    new_bg = bg * tf.expand_dims(intensity_mask, 2)
    tf.gfile.FastGFile(os.path.join('output', inp), new_bg)

