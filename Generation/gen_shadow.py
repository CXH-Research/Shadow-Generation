import utils as utils
import datasets as dts
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

# tf.compat.v1.enable_eager_execution(
#     config=None, device_policy=None, execution_mode=None
# )
TONE_SIGMA = 0.1
SS_SIGMA = 0.5

size = (256, 256)

bg_path = './bg/1-1.png'

intensity_mask = utils.get_brightness_mask(size=size, min_val=0.1)
bg = cv2.resize(cv2.imread(bg_path), (256, 256))
# bg = bg * tf.expand_dims(intensity_mask, 2)
bg = bg * np.expand_dims(intensity_mask.numpy(), 2)


cv2.imshow('mask', intensity_mask.numpy())
cv2.imshow('bg', bg * intensity_mask.numpy())
cv2.waitKey(0)