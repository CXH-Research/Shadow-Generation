import utils as utils
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import argparse
import os
import pandas as pd


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
parser.add_argument('--num_shadow', type=int, default=1,
                    help='Number of shadows')
args = parser.parse_args()

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

TONE_SIGMA = 0.1
SS_SIGMA = 0.5

size = (args.height, args.width)

inputs = os.listdir('input')
inputs.remove('.gitkeep')

columns = ['input', 'mask', 'output']
df = pd.DataFrame(columns=columns)

for inp in inputs:
    img_format = inp.split('.')[-1]
    if img_format == 'png':
        bg = utils.read_float(os.path.join('input', inp),
                              channel=3, itype='png', is_linear=False)
    else:
        bg = utils.read_float(os.path.join('input', inp),
                              channel=3, itype='jpg', is_linear=False)
    bg = cv2.resize(bg.numpy(), (args.width, args.height))
    # create mask
    for i in range(0, args.num_shadow):
        min_val = random.uniform(args.min_val, 1)
        intensity_mask = utils.get_brightness_mask(size=size, min_val=0.7)
        shadow = bg * tf.expand_dims(intensity_mask, 2)
        
        filename = str(i) + '_' + inp
        save_image(intensity_mask, True, 'mask', filename)
        save_image(shadow, False, 'output', filename)
        row = {'input': os.path.join('input', inp), 'mask': os.path.join('mask', filename), 'output': os.path.join('output', filename)}
        df = pd.concat([df, pd.DataFrame([row])])
        df = df[columns]

df.to_csv('./label.csv', encoding='utf-8', index=False)
