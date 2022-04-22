from turtle import back
import cv2
import numpy as np
import argparse
import random
import os

parser = argparse.ArgumentParser(description='This is the composition method to generate shadow')
parser.add_argument('--alpha_min', type=int, default=0.4, help='Minimum alpha value')
parser.add_argument('--alpha_max', type=int, default=0.7, help='Maximum alpha value')
parser.add_argument('--height', type=int, default=256,
                    help='Image height')
parser.add_argument('--width', type=int, default=256,
                    help='Image width')
parser.add_argument('--mode', type=str, default='align',
                    help='Minimum alpha value', choices=['align', 'random'])
args = parser.parse_args()

height = args.height
width = args.width

inputs = os.listdir('input')
masks = os.listdir('mask')

for inp in inputs:
    inp_img = cv2.imread(os.path.join('input', inp))
    inp_img = cv2.resize(inp_img, (width, height))
    if args.mode == 'align':
        mask_img = cv2.imread(os.path.join('mask', inp))
    else:
        mask_img = cv2.imread(os.path.join('mask', random.choice(masks)))
    mask_img = cv2.resize(mask_img, (width, height))
    shadow = 255 - mask_img
    output = np.zeros((height, width, 3))
    alpha = random.uniform(args.alpha_min, args.alpha_max)
    for h in range(0, height):
        for w in range(0, width):
            if shadow[h][w][0] == 255:
                output[h][w] = inp_img[h][w]
            else:
                output[h][w] = alpha * inp_img[h][w] + \
                    (1 - alpha) * shadow[h][w]
    cv2.imwrite(os.path.join('output', inp), output)
