__author__ = 'kai'
# ! -*- coding: UTF-8 -*-


import numpy as np
import cv2
from common import mulShow
import matplotlib.pyplot as plt
import skimage.io as skio
from gradientMag import gradientMag


ORIENTATION_DEGREES = [15, 45, 75, 105, 135, 165]  # Central degrees for gradient orientation channels.
ORIENTATION_BIN = 14  # Bin size for orientation channels.


def oriented_gradient(degree_mat, degree, bin_size):

    lower_bound = degree - bin_size
    upper_bound = degree + bin_size

    rows, cols = degree_mat.shape

    oriented = np.zeros((rows, cols), np.uint8)

    mask = (degree_mat > lower_bound) * (degree_mat < upper_bound)
    oriented[mask] = 255

    return oriented


def gradHistChnCompute(M, O):
    assert len(ORIENTATION_DEGREES) == 6
    assert min(ORIENTATION_DEGREES) - ORIENTATION_BIN > 0
    assert max(ORIENTATION_DEGREES) + ORIENTATION_BIN < 180

    degree = O * 180 / np.pi

    for i, deg in enumerate(ORIENTATION_DEGREES):
        orie = oriented_gradient(degree, deg, ORIENTATION_BIN)
        orie = cv2.medianBlur(orie, 3)
        orie = cv2.bitwise_and(orie, M)

        H = orie.copy() if not i else np.dstack((H, orie))

    return H


if __name__ == '__main__':
    I = skio.imread('peppers.png')
    M, O = gradientMag(I)
    mulShow(M)
    # H = gradHistChnCompute(M, O)
    # mulShow(H)
    print 'test'
