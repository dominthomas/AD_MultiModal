import numpy as np
import tensorflow as tf
from sklearn import metrics
from Class_Activation_Maps import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import argparse
import imutils
import cv2
import os

import os
import re
import gc
import cv2
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
        angles = [1, -1, 2, -2, -3, 3, 1.5, -1.5]

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
with open('ad_sub_test_files', 'r') as filehandle:
    ad_sub_test_files = [current_file.rstrip() for current_file in filehandle.readlines()]
with open('cn_sub_test_files', 'r') as filehandle:
    cn_sub_test_files = [current_file.rstrip() for current_file in filehandle.readlines()]

slices_s = [60, 61, 62]
slices_a = [80, 81, 82]

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/AD/")
ad_test_s = get_images(ad_sub_test_files, plane="s", slices=slices_s)
ad_test_c = get_images(ad_sub_test_files, plane="c")
ad_test_a = get_images(ad_sub_test_files, plane="a", slices=slices_a)

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/CN/")
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
os.chdir('/home/dthomas')

model_sagittal = tf.keras.models.load_model('sagittal_60_61_62')
model_coronal = tf.keras.models.load_model('coronal_86_87_88')
model_axial = tf.keras.models.load_model('axial_80_81_82')

pred_s = model_sagittal.predict_classes(test_s).flatten()
pred_c = model_coronal.predict_classes(test_c).flatten()
pred_a = model_axial.predict_classes(test_a).flatten()

# print(model_sagittal.evaluate(test_s, test_labels))
# print(model_coronal.evaluate(test_c, test_labels))
# print(model_axial.evaluate(test_a, test_labels))

pred_sum = pred_s + pred_c + pred_a
print(pred_sum)
pred_sum = [1 if x >= 1 else 0 for x in pred_sum]
print(pred_sum)
report = metrics.classification_report(test_labels, pred_sum)
print(report)
print(metrics.balanced_accuracy_score(test_labels, pred_sum))
#################################################

# image = cv2.imread('/home/dthomas/AD/2D_MultiModal/OASIS3/AD/OAS30782_d0074run-02/c/118.png')
# print(image.shape)
image = test_c[0]
image = np.expand_dims(image, axis=0)
preds = model_coronal.predict(image)
pred_class = model_coronal.predict_classes(image)
i = np.argmax(preds[0])
image_colour = np.stack((test_c[0],) * 3, axis=2)
image_colour = image_colour[:, :, :, 0]
#image_colour = imagenet_utils.preprocess_input(image_colour)
print(image_colour.shape)

print("Prediction vs Ground Truth:")
print(pred_class)
print(test_labels[0])

cam = GradCAM(model_coronal, i)
heatmap = cam.compute_heatmap(image)

heatmap = cv2.resize(heatmap, (227, 227))
print("Heatmap shape", heatmap.shape)
print("image shape", image_colour.shape)
(heatmap, output) = cam.overlay_heatmap(heatmap, image_colour.astype('uint8'), alpha=0.5)

# cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
#cv2.putText(output, test_labels[0], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#            0.8, (255, 255, 255), 2)

print(output.shape)

print(type(output[:, 1, 0]))
print(type(heatmap))
print(type(image_colour))

output = np.vstack([image_colour, heatmap, output])
# output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)
