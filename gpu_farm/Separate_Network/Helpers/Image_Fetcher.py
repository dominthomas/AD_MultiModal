import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import re

"""@author Domin Thomas"""

"""Crop function"""


def crop(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
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

            # return_list = return_list + get_noisy_images(png0)
            # return_list = return_list + get_noisy_images(png1)
            # return_list = return_list + get_noisy_images(png2)

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
