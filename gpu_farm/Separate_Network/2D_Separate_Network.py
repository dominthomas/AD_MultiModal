import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
import multiprocessing as mp
from itertools import cycle, chain

import os
import re
import gc
import shutil

from gpu_farm.Separate_Network.Image_Fetcher import get_images
from gpu_farm.Separate_Network.Test_Helper import test_model

"""@author Domin Thomas"""
"""Configure GPUs to prevent OOM errors"""

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""Set random seed for reproducibility"""
tf.random.set_seed(129)

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

accuracies = []

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
        pool = mp.Pool(32)
        ad_train = pool.starmap(get_images, [([file], plane, slices, True, True) for file in ad_sub_train_files])
        pool.close()
        ad_train = list(chain.from_iterable(ad_train))

        """Retrieve CN Files"""
        print("Retrieving CN Train in the plane ", plane)
        os.chdir("/home/k1651915/2D_MultiModal/OASIS3/CN/")
        pool = mp.Pool(32)
        cn_train = pool.starmap(get_images, [([file], plane, slices, True) for file in cn_sub_train_files])
        pool.close()
        cn_train = list(chain.from_iterable(cn_train))

        train = np.asarray(cn_train + ad_train)

        y1 = np.zeros(len(cn_train))
        y2 = np.ones(len(ad_train))
        train_labels = np.concatenate((y1, y2), axis=None)

        print(len(cn_train))
        print(len(ad_train))

        cn_train = None
        ad_train = None
        gc.collect()

        #################################################
        with tf.device("/cpu:0"):
            with tf.device("/gpu:0"):
                model = tf.keras.Sequential()

                model.add(Conv2D(32,
                                 input_shape=(227, 227, 1),
                                 data_format='channels_last',
                                 kernel_size=(7, 7),
                                 strides=(4, 4),
                                 padding='valid',
                                 activation='relu'))

            with tf.device("/gpu:1"):
                model.add(MaxPooling2D(pool_size=(2, 2),
                                       strides=(2, 2),
                                       padding='valid'))

            with tf.device("/gpu:2"):
                model.add(Conv2D(64,
                                 kernel_size=(5, 5),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu'))

                model.add(MaxPooling2D(pool_size=(2, 2),
                                       strides=(2, 2),
                                       padding='valid'))
            with tf.device("/gpu:3"):
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

            with tf.device("/gpu:4"):
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

                model.add(Flatten())

                model.add(Dense(32, activation='relu'))

                model.add(Dense(1, activation='sigmoid'))
        #################################################

        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(train,
                  train_labels,
                  epochs=20,
                  batch_size=512,
                  shuffle=True)
        #################################################

        model_file_name = "coronal_86_87_88"

        if plane == "s":
            model_file_name = "sagittal_60_61_62"
        elif plane == "a":
            model_file_name = "axial_80_81_82"

        os.chdir('/home/k1651915/')
        model.save(model_file_name)

        K.clear_session()
        ad_train = None
        cn_train = None
        gc.collect()

    """Test the model"""
    test_model(ad_sub_test_files, cn_sub_test_files)

    # Remove models
    os.chdir('/home/k1651915/')
    shutil.rmtree("coronal_86_87_88")
    shutil.rmtree("sagittal_60_61_62")
    shutil.rmtree("axial_80_81_82")
