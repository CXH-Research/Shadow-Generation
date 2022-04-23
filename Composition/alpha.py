import argparse
from sys import argv
import random
import os
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

def save_shadow(inp_img, mask_img, filename):
    foremask = 1 - mask_img
    alpha = random.uniform(args.alpha_min, args.alpha_max)
    output = inp_img * foremask + (inp_img * mask_img * alpha + foremask * (1 - alpha)) * mask_img
    save_image(output, os.path.join('output', filename))

mode = ['align', 'random']

parser = argparse.ArgumentParser(description='This is the composition method to generate shadow')
parser.add_argument('--alpha_min', type=int, default=0.2, help='Minimum alpha value')
parser.add_argument('--alpha_max', type=int, default=0.7, help='Maximum alpha value')
parser.add_argument('--height', type=int, default=1754,
                    help='Image height')
parser.add_argument('--width', type=int, default=1240,
                    help='Image width')
parser.add_argument('--mode', type=str, default='align',
                    help='Minimum alpha value', choices=['align', 'random'])
parser.add_argument('--num_shadow', type=int, default=1,
                    help='Number of shadows', required=(mode[1] in argv))
args = parser.parse_args()

height = args.height
width = args.width

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = os.listdir('input')
masks = os.listdir('mask')
inputs.remove('.gitkeep')
masks.remove('.gitkeep')

columns = ['input', 'mask', 'output']
df = pd.DataFrame(columns=columns)

for inp in tqdm(inputs):
    inp_img = Image.open(os.path.join('input', inp))
    inp_img = TF.to_tensor(inp_img).to(device)
    inp_img = TF.resize(inp_img, [height, width])
    save_image(inp_img, os.path.join('input', inp))
    if args.mode == 'align':
        mask_img = Image.open(os.path.join('mask', inp))
        mask_img = TF.to_tensor(mask_img).to(device)
        mask_img = TF.resize(mask_img, [height, width])
        save_shadow(inp_img, mask_img, inp)
        row = {'input': os.path.join('input', inp), 'mask': os.path.join('mask', inp), 'output': os.path.join('output', inp)}
    else:
        sampled_shadows = random.sample(masks, args.num_shadow)
        cnt = 1
        for shadow in sampled_shadows:
            mask_img = Image.open(os.path.join('mask', shadow))
            mask_img = TF.to_tensor(mask_img).to(device)
            mask_img = TF.resize(mask_img, [height, width])
            filename = str(cnt) + '_shadow_' + inp
            save_shadow(inp_img, mask_img, filename)
            cnt += 1
            row = {'input': os.path.join('input', inp), 'mask': os.path.join('mask', shadow), 'output': os.path.join('output', filename)}
    df = pd.concat([df, pd.DataFrame([row])])
    df = df[columns]

df.to_csv('./label.csv', encoding='utf-8', index=False)


