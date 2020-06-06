import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D

import os
import re
import gc
import cv2
import numpy as np
import matplotlib.image as mpimg


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
        angles = [1, -1, 2, -2, -3, 3, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 4, -4]
    else:
        angles = [1, -1]

    rotated_pngs = []

    if custom_angle:
        angles = [angle]

    (h, w) = png.shape[:2]
    center = (w / 2, h / 2)

    for angle in angles:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        r = cv2.warpAffine(png, m, (h, w))
        r = cv2.resize(crop(r), (227, 227))
        r = np.stack((r,) * 1, axis=2)
        rotated_pngs.append(r)

    return rotated_pngs


os.chdir('/home/dthomas')
with open('ad_sub_train_files', 'r') as filehandle:
    ad_sub_train_files = [current_file.rstrip() for current_file in filehandle.readlines()]
with open('cn_sub_train_files', 'r') as filehandle:
    cn_sub_train_files = [current_file.rstrip() for current_file in filehandle.readlines()]

all_planes = ["s", "c", "a"]

for plane in all_planes:
    slices = [86, 87, 88]

    if plane == "s":
        slices = [60, 61, 62]
    elif plane == "a":
        slices = [80, 81, 82]

    os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/AD/")
    ad_train = get_images(ad_sub_train_files, train=True, plane=plane, slices=slices, ad=True)

    os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/CN/")
    cn_train = get_images(cn_sub_train_files, train=True, plane=plane, slices=slices)

    train_s = np.asarray(cn_train + ad_train)

    y1 = np.zeros(len(cn_train))
    y2 = np.ones(len(ad_train))
    train_labels = np.concatenate((y1, y2), axis=None)

    print(len(cn_train))
    print(len(ad_train))

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

    #################################################
    model = tf.keras.Sequential()

    model.add(Conv2D(32,
                     input_shape=(227, 227, 1),
                     data_format='channels_last',
                     kernel_size=(7, 7),
                     strides=(4, 4),
                     padding='valid',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(64,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))

    model.add(Conv2D(384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))

    model.add(Conv2D(384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))

    model.add(Conv2D(512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))

    model.add(Conv2D(256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid'))

    model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    #################################################

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_s,
              train_labels,
              epochs=10,
              batch_size=512,
              shuffle=True)
    #################################################

    model_file_name = "coronal_86_87_88"

    if plane == "s":
        model_file_name = "sagittal_60_61_62"
    elif plane == "a":
        model_file_name = "axial_80_81_82"

    os.chdir('/home/dthomas/')
    model.save(model_file_name)
    gc.collect()

