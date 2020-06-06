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

import os
import re
import gc
import random
import cv2
import matplotlib.image as mpimg

"""@author Domin Thomas"""


def crop(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


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


ad_files = os.listdir("/home/k1651915/2D_MultiModal/OASIS3/AD/")
cn_files = os.listdir("/home/k1651915/2D_MultiModal/OASIS3/CN/")

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

slices_s = [60, 61, 62]
slices_a = [80, 81, 82]


