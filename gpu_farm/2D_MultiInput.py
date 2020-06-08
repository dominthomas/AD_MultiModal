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
from sklearn.model_selection import KFold

import os
import re
import gc
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
        angles = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]
    else:
        angles = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]

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

kf = KFold(n_splits=15)
kf_ad_sub_id = kf.split(sub_id_ad)
kf_cn_sub_id = kf.split(sub_id_cn)

train_indexes_ad = []
test_indexes_ad = []

for train_index_ad, test_index_ad in kf_ad_sub_id:
    train_indexes_ad.append(list(train_index_ad))
    test_indexes_ad.append(list(test_index_ad))

count = 0
accuracies = []

for train_index_cn, test_index_cn in kf_cn_sub_id:
    cn_train_subs = [sub_id_cn[i] for i in train_index_cn]
    cn_test_subs = [sub_id_cn[i] for i in test_index_cn]

    ad_train_subs = [sub_id_ad[i] for i in train_indexes_ad[count]]
    ad_test_subs = [sub_id_ad[i] for i in test_indexes_ad[count]]

    ad_sub_train_files = []
    ad_sub_test_files = []

    cn_sub_train_files = []
    cn_sub_test_files = []

    for file in ad_files:
        file_sub_id = re.search('(OAS\\d*)', file).group(1)
        if file_sub_id in ad_train_subs:
            ad_sub_train_files.append(file)

    for file in cn_files:
        file_sub_id = re.search('(OAS\\d*)', file).group(1)
        if file_sub_id in cn_train_subs:
            cn_sub_train_files.append(file)

    for ad_sub in ad_test_subs:
        r = re.compile(ad_sub)
        files = list(filter(r.match, ad_files))
        ad_sub_test_files.append(files[0])

    for cn_sub in cn_test_subs:
        r = re.compile(cn_sub)
        files = list(filter(r.match, cn_files))
        cn_sub_test_files.append(files[0])

    data_length = len(ad_sub_test_files)

    os.chdir("/home/k1651915/2D_MultiModal/OASIS3/AD/")
    ad_train_s = get_images(ad_sub_train_files, train=True, ad=True, plane="s", slices=slices_s)
    ad_train_c = get_images(ad_sub_train_files, train=True, ad=True, plane="c")
    ad_train_a = get_images(ad_sub_train_files, train=True, ad=True, plane="a", slices=slices_a)

    ad_test_s = get_images(ad_sub_test_files, plane="s", slices=slices_s, same_length=True, data_length=data_length)
    ad_test_c = get_images(ad_sub_test_files, plane="c", same_length=True, data_length=data_length)
    ad_test_a = get_images(ad_sub_test_files, plane="a", slices=slices_a, same_length=True, data_length=data_length)

    os.chdir("/home/k1651915/2D_MultiModal/OASIS3/CN/")
    cn_train_s = get_images(cn_sub_train_files, train=True, plane="s", slices=slices_s)
    cn_train_c = get_images(cn_sub_train_files, train=True, plane="c")
    cn_train_a = get_images(cn_sub_train_files, train=True, plane="a", slices=slices_a)

    cn_test_s = get_images(cn_sub_test_files, plane="s", slices=slices_s, same_length=True, data_length=data_length)
    cn_test_c = get_images(cn_sub_test_files, plane="c", same_length=True, data_length=data_length)
    cn_test_a = get_images(cn_sub_test_files, plane="a", slices=slices_a, same_length=True, data_length=data_length)

    train_s = np.asarray(cn_train_s + ad_train_s)
    train_c = np.asarray(cn_train_c + ad_train_c)
    train_a = np.asarray(cn_train_a + ad_train_a)

    test_s = np.asarray(cn_test_s + ad_test_s)
    test_c = np.asarray(cn_test_c + ad_test_c)
    test_a = np.asarray(cn_test_a + ad_test_a)

    y1 = np.zeros(len(cn_train_s))
    y2 = np.ones(len(ad_train_s))
    train_labels = np.concatenate((y1, y2), axis=None)

    y1 = np.zeros(len(cn_test_s))
    y2 = np.ones(len(ad_test_s))
    test_labels = np.concatenate((y1, y2), axis=None)

    print(len(test_s))
    print(len(train_s))

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

    input_s = Input(shape=(227, 227, 1))

    c1 = Conv2D(32,
                input_shape=(227, 227, 1),
                data_format='channels_last',
                kernel_size=(7, 7),
                strides=(4, 4),
                padding='valid',
                activation='relu')(input_s)

    mp1 = MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2),
                       padding='valid')(c1)

    c2 = Conv2D(64,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding='valid',
                activation='relu')(mp1)

    mp2 = MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2),
                       padding='valid')(c2)

    c3 = Conv2D(384,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='valid',
                activation='relu')(mp2)

    c4 = Conv2D(384,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='valid',
                activation='relu')(c3)

    c5 = Conv2D(512,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='valid',
                activation='relu')(c4)

    c6 = Conv2D(256,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='valid',
                activation='relu')(c5)

    mp3 = MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2),
                       padding='valid')(c6)

    flat1 = Flatten()(mp3)

    input_c = Input(shape=(227, 227, 1))

    c1_c = Conv2D(32,
                  input_shape=(227, 227, 1),
                  data_format='channels_last',
                  kernel_size=(7, 7),
                  strides=(4, 4),
                  padding='valid',
                  activation='relu')(input_c)

    mp1_c = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='valid')(c1_c)

    c2_c = Conv2D(64,
                  kernel_size=(5, 5),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(mp1_c)

    mp2_c = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='valid')(c2_c)

    c3_c = Conv2D(384,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(mp2_c)

    c4_c = Conv2D(384,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(c3_c)

    c5_c = Conv2D(512,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(c4_c)

    c6_c = Conv2D(256,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(c5_c)

    mp3_c = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='valid')(c6_c)

    flat2 = Flatten()(mp3_c)

    input_a = Input(shape=(227, 227, 1))

    c1_a = Conv2D(32,
                  input_shape=(227, 227, 1),
                  data_format='channels_last',
                  kernel_size=(7, 7),
                  strides=(4, 4),
                  padding='valid',
                  activation='relu')(input_a)

    mp1_a = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='valid')(c1_a)

    c2_a = Conv2D(64,
                  kernel_size=(5, 5),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(mp1_a)

    mp2_a = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='valid')(c2_a)

    c3_a = Conv2D(384,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(mp2_a)

    c4_a = Conv2D(384,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(c3_a)

    c5_a = Conv2D(512,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(c4_a)

    c6_a = Conv2D(256,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='valid',
                  activation='relu')(c5_a)

    mp3_a = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='valid')(c6_a)

    flat3 = Flatten()(mp3_a)

    merge = concatenate([flat1, flat2, flat3])

    gc.collect()

    h1 = Dense(32, activation='relu')(merge)
    output = Dense(1, activation='sigmoid')(h1)

    model = Model(inputs=[input_s, input_c, input_a],
                  outputs=output)

    gc.collect()

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    gc.collect()

    model.fit([train_s, train_c, train_a],
              train_labels,
              epochs=20,
              batch_size=120,
              shuffle=True)
    #################################################

    gc.collect()
    evaluation = model.evaluate([test_s, test_c, test_a], test_labels, verbose=0)
    print(evaluation)
    accuracies.append(evaluation[1])
    print(np.mean(accuracies))
