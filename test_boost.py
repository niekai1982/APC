import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from gradientMag import gradientMag_opencv
from time import time

def read_sampleMat(path):
    sample = sio.loadmat(path)
    return sample["Is1"]


def get_feature(I, shrink):
    src = I.astype(np.float32) * 1. / 255
    h, w = src.shape[:2]

    cr = np.mod([h, w], shrink)
    if np.any(cr):
        h -= cr[0]
        w -= cr[1]

    src = src[:h, :w]

    luv = cv2.cvtColor(src, cv2.COLOR_RGB2LUV)
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    row_der = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    col_der = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    grad, angles = cv2.cartToPolar(col_der, row_der, angleInDegrees=True)
    angles[angles > 180] -= 180
    hist = np.zeros((I.shape[0], I.shape[1], 6), dtype=np.float32)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            angle = angles[i, j]
            ind = (int)(angle / 30)
            if ind==6:
                ind = 5
            hist[i,j,ind] = grad[i,j] * 255

    feature = luv.copy()
    feature = np.dstack((luv, grad))
    feature = np.dstack((feature, hist))
    return feature


if __name__ == '__main__':
    sample_path = "H:/pos_sample.mat"
    sample = read_sampleMat(sample_path)
    sp_t = sample[:,:,:,0]

    # sp_t = cv2.imread("C:/peppers.png")
    # sp_t = sp_t[:,:,::-1]

    start = time()
    features = get_feature(sp_t, shrink=4)
    end = time()
    print end - start
