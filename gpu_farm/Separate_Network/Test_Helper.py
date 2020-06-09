import numpy as np
import tensorflow as tf
from sklearn import metrics

import os
import gc

from gpu_farm.Separate_Network.Image_Fetcher import get_images

"""@author Domin Thomas"""
slices_s = [60, 61, 62]
slices_a = [80, 81, 82]


def test_model(ad_sub_test_files, cn_sub_test_files):
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
    print(len(cn_test_c))
    print(len(cn_test_a))

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
    pred_sum = [1 if x >= 1 else 0 for x in pred_sum]
    report = metrics.classification_report(test_labels, pred_sum)
    print(report)
    print(metrics.balanced_accuracy_score(test_labels, pred_sum))
    gc.collect()
    #################################################




