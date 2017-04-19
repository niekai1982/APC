__author__ = 'kai'
# ! -*- coding: UTF-8 -*-

import numpy as np
import cv2
from rgbConvert import rgbConvert
import scipy.io as sio
import skimage.io as skio
from common import mulShow


def gradientMag(I):
    gray = rgbConvert(I, 'gray')
    Gx, Gy = np.gradient(gray)
    M = np.sqrt(np.square(Gx) + np.square(Gy))
    O = np.arctan2(Gy, Gx)
    return M, O


def gradientMag_opencv(I):
    src = I.astype(np.float32) * 1. / 255
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    row_der = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    col_der = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    M, O = cv2.cartToPolar(col_der, row_der, angleInDegrees=True)
    return M, O


if __name__ == '__main__':
    print cv2.__version__
    I = cv2.imread("c:\peppers.png")
    # I = skio.imread('C:\peppers.png')
    M, O = gradientMag_opencv(I)
    mulShow(I, M, O)
