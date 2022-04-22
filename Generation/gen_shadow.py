import utils as utils
import datasets as dts
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
TONE_SIGMA = 0.1
SS_SIGMA = 0.5

size = (256, 256)
intensity_mask = utils.get_brightness_mask(size=size, min_val=0.4)

bg_path = './bg/1-1.png'
silhouette_path = './silhouette/90-1.png'
segmentation_path = './input_mask_ind/008_0.png'
image_path = './input/008.png'
bbox_path = './input_mask_ind/008.txt'
bbox = dts.read_bbox(bbox_path)[0]

data_dict = {}
data_dict['bbox'] = bbox
data_dict['bg'] = utils.read_float(
    bg_path, channel=3, itype='png', is_linear=True)
data_dict['silhouette'] = utils.read_float(
    silhouette_path, channel=1, itype='png', is_linear=False)
data_dict['segmentation'] = utils.read_float(
    segmentation_path, channel=1, itype='png', is_linear=False)
data_dict['shadowed_before'] = utils.read_float(
    image_path, channel=3, itype='jpg', is_linear=True)


rsz_ratio = 1.0
bbox = dts.read_bbox(bbox_path)[0]
image_concat = tf.concat(
    [data_dict['shadowed_before'], data_dict['segmentation']], axis=2)
processed_images = dts.align_images_and_segmentation(
    image_concat, size=size, bbox=bbox, rsz=rsz_ratio, param_save=False, is_train=True)
shadow_lit_image = processed_images[..., :3]
segmentation = processed_images[..., -1:]

shadow_mask_hard = 1 - data_dict['silhouette']
shadow_mask_hard_perlin = utils.render_perlin_mask(size=size)
shadow_mask_hard_silh = utils.render_silhouette_mask(
    silhouette=shadow_mask_hard,
    size=size,
    segmentation=segmentation)

shadow_mask_hard = shadow_mask_hard_perlin
shadow_mask_hard_inv = 1 - shadow_mask_hard

prob_apply_ss = tf.random.uniform([])
shadow_mask_ss = tf.cond(
    tf.greater(prob_apply_ss, tf.constant(SS_SIGMA)),
    lambda: utils.apply_ss_shadow_map(
        shadow_mask_hard),
    lambda: tf.image.grayscale_to_rgb(shadow_mask_hard))
shadow_mask_ss_inv = 1 - shadow_mask_ss

intensity_mask = utils.get_brightness_mask(size=size, min_val=0.1)
shadow_mask_sv = shadow_mask_ss_inv * tf.expand_dims(intensity_mask, 2)

bg = cv2.resize(np.array(data_dict['bg']), (256, 256))
