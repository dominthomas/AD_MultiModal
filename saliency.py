import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model
import PIL
from vis.visualization import visualize_saliency
from vis.utils import utils

import matplotlib.pyplot as plt
from tensorflow.keras import activations
import matplotlib.image as mpimage
import scipy.ndimage as ndimage

import os
import re
import gc
import random
import cv2
import matplotlib.image as mpimg
import time

ad_files = os.listdir("/home/dthomas/AD/2D_MultiModal/OASIS3/AD/")
cn_files = os.listdir("/home/dthomas/AD/2D_MultiModal/OASIS3/CN/")

sub_id_ad = []
sub_id_cn = []
for file in ad_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_ad:
        sub_id_ad.append(sub_id)

for file in cn_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_cn:
        sub_id_cn.append(sub_id)


def crop(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def get_images(folders, plane, slices=None, train=False, same_length=False, data_length=0, adni=False, ad=False):
    if slices is None:
        slices = [86, 87, 88]

    data_length = data_length * 3

    return_list = []
    for folder in folders:

        if same_length and len(return_list) == data_length:
            return return_list

        file_num_only = []
        os.chdir(folder)
        files = os.listdir(str(plane))

        for png_file in files:
            file_num_only.append(int(re.search('(\\d*)', png_file).group(1)))

        file_num_only.sort()
        png0 = mpimg.imread(str(plane) + "/" + str(file_num_only[slices[0]]) + ".png")
        png1 = mpimg.imread(str(plane) + "/" + str(file_num_only[slices[1]]) + ".png")
        png2 = mpimg.imread(str(plane) + "/" + str(file_num_only[slices[2]]) + ".png")

        png0 = png0[:, :, 1]
        png1 = png1[:, :, 1]
        png2 = png2[:, :, 1]

        if adni:
            png0 = get_rotated_images(png0, custom_angle=True, angle=-90)[0][:, :, 0]
            png1 = get_rotated_images(png1, custom_angle=True, angle=-90)[0][:, :, 0]
            png2 = get_rotated_images(png2, custom_angle=True, angle=-90)[0][:, :, 0]

        if train:
            return_list = return_list + get_rotated_images(png0, ad=ad)
            return_list = return_list + get_rotated_images(png1, ad=ad)
            return_list = return_list + get_rotated_images(png2, ad=ad)

        png0 = crop(png0)
        png1 = crop(png1)
        png2 = crop(png2)

        png0 = cv2.resize(png0, (227, 227))
        png1 = cv2.resize(png1, (227, 227))
        png2 = cv2.resize(png2, (227, 227))

        png0 = np.stack((png0,) * 1, axis=2)
        png1 = np.stack((png1,) * 1, axis=2)
        png2 = np.stack((png2,) * 1, axis=2)

        return_list = return_list + [png0, png1, png2]
        os.chdir('../')

    return return_list


def get_rotated_images(png, custom_angle=False, angle=0, ad=False):
    if ad:
        angles = [3, -4, -3, 4, 1, 2, -1, -2, -0.5, 0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5]
    else:
        angles = [1, -1]

    rotated_pngs = []

    if custom_angle:
        angles = [angle]
    else:
        png = get_rotated_images(png, True, -90)[0][:, :, 0]

    (h, w) = png.shape[:2]
    center = (w / 2, h / 2)

    for angle in angles:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        r = cv2.warpAffine(png, m, (h, w))
        r = cv2.resize(crop(r), (227, 227))
        r = np.stack((r,) * 1, axis=2)
        rotated_pngs.append(r)

    return rotated_pngs


results = []
seeds = range(1, 10)

random.seed(1)

random.shuffle(sub_id_ad)
random.shuffle(sub_id_cn)

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/AD/")
ad_sub_train = sub_id_ad[0:164]
ad_sub_test = sub_id_ad[165:177]

ad_sub_train_files = []
ad_sub_validate_files = []
ad_sub_test_files = []

for file in ad_files:
    file_sub_id = re.search('(OAS\\d*)', file).group(1)
    if file_sub_id in ad_sub_train:
        ad_sub_train_files.append(file)
    elif file_sub_id in ad_sub_test:
        ad_sub_test_files.append(file)

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/CN")
cn_sub_train = sub_id_cn[0:574]
cn_sub_test = sub_id_cn[575:587]

cn_sub_train_files = []
cn_sub_test_files = []

for file in cn_files:
    file_sub_id = re.search('(OAS\\d*)', file).group(1)
    if file_sub_id in cn_sub_train:
        cn_sub_train_files.append(file)
    elif file_sub_id in cn_sub_test:
        cn_sub_test_files.append(file)

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/AD/")
ad_test_c = get_images(ad_sub_test_files, plane="c", same_length=True, data_length=12)

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/CN/")
cn_test_c = get_images(cn_sub_test_files, plane="c", same_length=True, data_length=12)


y1 = np.zeros(len(cn_test_c))
y2 = np.ones(len(ad_test_c))
test_labels = np.concatenate((y1, y2), axis=None)

cn_train_s = None
cn_train_c = None
cn_train_a = None

ad_train_s = None
ad_train_c = None
ad_train_a = None

cn_test_s = None
cn_test_c = None
cn_test_a = None

ad_test_s = None
ad_test_c = None
ad_test_a = None

gc.collect()

os.chdir('/home/dthomas')
coronal_model = load_model('coronal')
coronal_model.summary()
#################################################
layer_idx = utils.find_layer_idx(coronal_model, 'dense_1')
coronal_model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(coronal_model)

grads = visualize_saliency(model, layer_idx, filter_indices=None,
                           seed_input= ad_test_c[0], backprop_modifier=None,
                           grad_modifier="absolute")
plt.imshow(grads, alpha=.6)
#################################################
time.sleep(10)
