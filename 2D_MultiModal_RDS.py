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
from sklearn import metrics
from sklearn.model_selection import KFold

import os
import re
import gc
import random
import cv2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()
readRDS = robjects.r['readRDS']

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
        slices = [87, 88, 89]

    data_length = data_length * 3

    return_list = []
    for folder in folders:

        if same_length and len(return_list) == data_length:
            return return_list

        os.chdir(folder)

        png0 = readRDS(str(plane) + "/" + str(slices[0]) + ".rds")
        png1 = readRDS(str(plane) + "/" + str(slices[1]) + ".rds")
        png2 = readRDS(str(plane) + "/" + str(slices[2]) + ".rds")

        if adni:
            png0 = get_rotated_images(png0, custom_angle=True, angle=-90)[0][:, :, 0]
            png1 = get_rotated_images(png1, custom_angle=True, angle=-90)[0][:, :, 0]
            png2 = get_rotated_images(png2, custom_angle=True, angle=-90)[0][:, :, 0]

        if train:
            return_list = return_list + get_rotated_images(png0, ad=ad)
            return_list = return_list + get_rotated_images(png1, ad=ad)
            return_list = return_list + get_rotated_images(png2, ad=ad)

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


kf = KFold(n_splits=15)
# kf.get_n_splits(sub_id_ad)

for train_index, test_index in kf.split(sub_id_ad):
    print("Train:", len(train_index), "Test:", len(test_index))
    ad_sub_train = [sub_id_ad[i] for i in train_index]
    ad_sub_test = [sub_id_ad[i] for i in test_index]

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

os.chdir("/home/dthomas/AD/RDS/OASIS3/AD/")
ad_train_s = get_images(ad_sub_train_files, plane="s", slices=[69, 70, 71])
ad_train_c = get_images(ad_sub_train_files, plane="c")
ad_train_a = get_images(ad_sub_train_files, plane="a", slices=[89, 90, 91])

ad_test_s = get_images(ad_sub_test_files, plane="s", slices=[69, 70, 71], same_length=True, data_length=12)
ad_test_c = get_images(ad_sub_test_files, plane="c", same_length=True, data_length=12)
ad_test_a = get_images(ad_sub_test_files, plane="a", slices=[89, 90, 91], same_length=True, data_length=12)

os.chdir("/home/dthomas/AD/RDS/OASIS3/CN/")
cn_train_s = get_images(cn_sub_train_files, plane="s", slices=[69, 70, 71])
cn_train_c = get_images(cn_sub_train_files, plane="c")
cn_train_a = get_images(cn_sub_train_files, plane="a", slices=[89, 90, 91])

cn_test_s = get_images(cn_sub_test_files, plane="s", slices=[69, 70, 71], same_length=True, data_length=12)
cn_test_c = get_images(cn_sub_test_files, plane="c", same_length=True, data_length=12)
cn_test_a = get_images(cn_sub_test_files, plane="a", slices=[89, 90, 91], same_length=True, data_length=12)

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

pandas2ri.deactivate()

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

h1 = Dense(32, activation='relu')(merge)
output = Dense(1, activation='sigmoid')(h1)

model = Model(inputs=[input_s, input_c, input_a],
              outputs=output)

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit([train_s, train_c, train_a],
          train_labels,
          epochs=50,
          batch_size=512,
          shuffle=True)
#################################################

gc.collect()

evaluation = model.evaluate([test_s, test_c, test_a], test_labels, verbose=0)
predictions = model.predict([test_s, test_c, test_a])
predicted_classes = np.argmax(predictions, axis=1)
report = metrics.classification_report(test_labels, predicted_classes)
print(report)

print(evaluation)
