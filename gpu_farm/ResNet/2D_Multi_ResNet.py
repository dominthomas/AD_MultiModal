import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn import metrics
import multiprocessing as mp
from itertools import cycle, chain
import matplotlib.image as mpimg
import cv2

import os
import re
import gc
import shutil

"""@author Domin Thomas"""
accuracies = []


class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
                        reg=0.0001, bnEps=2e-5, bnMom=0.9):

        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)

        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False,
                              kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bnEps=2e-5, bnMom=0.9):

        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(inputs)

        # apply CONV => BN => ACT => POOL to reduce spatial size
        x = Conv2D(filters[0], (5, 5), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("sigmoid")(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model


"""Crop function"""


def crop(img, tol=0):
    # img is 2D image data
    # tol is tolerance
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


"""Rotate Images"""


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


"""Add Gaussian and Salt & Pepper noise"""


def get_noisy_images(png):
    png = cv2.resize(crop(png), (227, 227))
    return_list = []

    row, col = png.shape

    # Gaussian
    mean = 0
    var = 0.01
    sigma = var ** 0.85
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = png + gauss
    return_list.append(np.stack((noisy,) * 1, axis=2))

    # Salt and Pepper
    s_vs_p = 0.9
    amount = 0.004
    out = np.copy(png)

    # Salt
    num_salt = np.ceil(amount * png.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in png.shape]
    out[tuple(coords)] = 1

    # Pepper
    num_pepper = np.ceil(amount * png.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in png.shape]
    out[tuple(coords)] = 0
    return_list.append(np.stack((out,) * 1, axis=2))

    return return_list


"""Retrieve 2D images"""


def get_images(folders, plane, slices=None, train=False, ad=False, same_length=False, data_length=0, adni=False):
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

            return_list = return_list + get_noisy_images(png0)
            return_list = return_list + get_noisy_images(png1)
            return_list = return_list + get_noisy_images(png2)

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


def test_model(ad_sub_test_files, cn_sub_test_files):
    slices_s = [60, 61, 62]
    slices_a = [80, 81, 82]
    os.chdir("/home/k1651915/2D_MultiModal/OASIS3/AD/")
    ad_test_s = get_images(ad_sub_test_files, plane="s", slices=slices_s)
    ad_test_c = get_images(ad_sub_test_files, plane="c")
    ad_test_a = get_images(ad_sub_test_files, plane="a", slices=slices_a)

    os.chdir("/home/k1651915/2D_MultiModal/OASIS3/CN/")
    cn_test_s = get_images(cn_sub_test_files, plane="s", slices=slices_s, same_length=True,
                           data_length=int(len(ad_test_s) / 3))
    cn_test_c = get_images(cn_sub_test_files, plane="c", same_length=True, data_length=int(len(ad_test_s) / 3))
    cn_test_a = get_images(cn_sub_test_files, plane="a", slices=slices_a, same_length=True,
                           data_length=int(len(ad_test_s) / 3))

    print(len(ad_test_s))
    print(len(cn_test_s))

    test_s = np.asarray(cn_test_s + ad_test_s)
    test_c = np.asarray(cn_test_c + ad_test_c)
    test_a = np.asarray(cn_test_a + ad_test_a)

    y1 = np.zeros(len(cn_test_s))
    y2 = np.ones(len(ad_test_s))
    test_labels = np.concatenate((y1, y2), axis=None)

    #################################################
    os.chdir('/home/k1651915')

    model_sagittal = tf.keras.models.load_model('sagittal_60_61_62')
    model_coronal = tf.keras.models.load_model('coronal_86_87_88')
    model_axial = tf.keras.models.load_model('axial_80_81_82')

    pred_s = model_sagittal.predict_classes(test_s).flatten()
    pred_c = model_coronal.predict_classes(test_c).flatten()
    pred_a = model_axial.predict_classes(test_a).flatten()

    pred_sum = pred_s + pred_c + pred_a
    pred_sum = [1 if x >= 2 else 0 for x in pred_sum]
    report = metrics.classification_report(test_labels, pred_sum)
    print(report)
    print(metrics.balanced_accuracy_score(test_labels, pred_sum))
    accuracies.append(metrics.balanced_accuracy_score(test_labels, pred_sum))
    print("Fold number is ", len(accuracies))
    print("The average accuracy right now is ", np.mean(accuracies))
    print("/=/=/=/=/=/=/=/=/---------------------------/=/=/=/=/=/=/=/=/")
    del model_sagittal
    del model_coronal
    del model_axial
    del test_s
    del test_c
    del test_a
    del ad_test_s
    del ad_test_c
    del ad_test_a
    del cn_test_s
    del cn_test_c
    del cn_test_a
    del report
    K.clear_session()
    gc.collect()
    #################################################


"""Retrieve AD & CN Filenames"""
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

"""Create KFolds of 15 and 50 for AD and CN respectively"""
kf = KFold(n_splits=15)
kf_ad_sub_id = kf.split(sub_id_ad)
kf = KFold(n_splits=50)
kf_cn_sub_id = kf.split(sub_id_cn)

train_indexes_ad = []
test_indexes_ad = []

for train_index_ad, test_index_ad in kf_ad_sub_id:
    train_indexes_ad.append(list(train_index_ad))
    test_indexes_ad.append(list(test_index_ad))

"""Create an infinite iterator to cycle through the AD folds"""
train_indexes_ad_iterator = cycle(train_indexes_ad)
test_indexes_ad_iterator = cycle(test_indexes_ad)

for train_index_cn, test_index_cn in kf_cn_sub_id:

    cn_train_subs = [sub_id_cn[i] for i in train_index_cn]
    cn_test_subs = [sub_id_cn[i] for i in test_index_cn]

    ad_train_subs = [sub_id_ad[i] for i in next(train_indexes_ad_iterator)]
    ad_test_subs = [sub_id_ad[i] for i in next(test_indexes_ad_iterator)]

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

    planes = ["s", "c", "a"]

    """Begin training each model"""
    for plane in planes:
        slices = [86, 87, 88]

        if plane == "s":
            slices = [60, 61, 62]
        elif plane == "a":
            slices = [80, 81, 82]

        """Retrieve AD Files"""
        print("Retrieving AD Train in the plane ", plane)
        os.chdir("/home/k1651915/2D_MultiModal/OASIS3/AD/")
        pool = mp.Pool(32, maxtasksperchild=32)
        ad_train = pool.starmap(get_images, [([file], plane, slices, True, True) for file in ad_sub_train_files])
        pool.close()
        pool.join()
        del pool
        gc.collect()
        ad_train = list(chain.from_iterable(ad_train))

        """Retrieve CN Files"""
        print("Retrieving CN Train in the plane ", plane)
        os.chdir("/home/k1651915/2D_MultiModal/OASIS3/CN/")
        pool = mp.Pool(32, maxtasksperchild=32)
        cn_train = pool.starmap(get_images, [([file], plane, slices, True) for file in cn_sub_train_files])
        pool.close()
        pool.join()
        del pool
        gc.collect()
        cn_train = list(chain.from_iterable(cn_train))

        train = np.asarray(cn_train + ad_train)

        y1 = np.zeros(len(cn_train))
        y2 = np.ones(len(ad_train))
        train_labels = np.concatenate((y1, y2), axis=None)

        print(len(cn_train))
        print(len(ad_train))

        del cn_train
        del ad_train
        gc.collect()

        #################################################
        """Configure GPUs to prevent OOM errors"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        """Set random seed for reproducibility"""
        tf.random.set_seed(129)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = ResNet.build(227, 227, 1, 1, (3, 4, 6, 8), (32, 64, 128, 256, 512))

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.005),
                      metrics=['accuracy'])

        model.fit(train,
                  train_labels,
                  epochs=20,
                  batch_size=32,
                  shuffle=True)
        #################################################

        model_file_name = "coronal_86_87_88"

        if plane == "s":
            model_file_name = "sagittal_60_61_62"
        elif plane == "a":
            model_file_name = "axial_80_81_82"

        os.chdir('/home/k1651915/')
        model.save(model_file_name)
        del model
        del train
        del train_labels
        gc.collect()

    """Test the model"""
    test_model(ad_sub_test_files, cn_sub_test_files)
    gc.collect()

    # Remove models
    os.chdir('/home/k1651915/')
    shutil.rmtree("coronal_86_87_88")
    shutil.rmtree("sagittal_60_61_62")
    shutil.rmtree("axial_80_81_82")
    K.clear_session()
    gc.collect()
