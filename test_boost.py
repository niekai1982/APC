import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from time import time

def read_sampleMat(path):
    sample = sio.loadmat(path)
    return sample["Is1"]


def get_feature(I):

    #change src image type from uchar to float
    src = I.astype(np.float32) * 1. / 255

    # #crop src image so divisible by shrink and get target dimensions
    # h, w = src.shape[:2]
    # cr = np.mod([h, w], shrink)
    # if np.any(cr):
    #     h -= cr[0]
    #     w -= cr[1]
    # src = src[:h, :w]

    #get luv channel
    luv = cv2.cvtColor(src, cv2.COLOR_RGB2LUV)

    #get gradientMag channel
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    row_der = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    col_der = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    grad, angles = cv2.cartToPolar(col_der, row_der, angleInDegrees=True)

    #get gradientHist channel
    angles[angles > 180] -= 180
    hist = np.zeros((I.shape[0], I.shape[1], 6), dtype=np.float32)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            angle = angles[i, j]
            ind = (int)(angle / 30)
            if ind==6:
                ind = 5
            hist[i,j,ind] = grad[i,j] * 255

    features = np.dstack((luv, grad, hist))
    return features


if __name__ == '__main__':
    sample_path = "H:/pos_sample.mat"
    sample = read_sampleMat(sample_path)
    sp_t = sample[:,:,:,0]

    features = get_feature(sp_t)
    print features.shape

    # sp_t = cv2.imread("C:/peppers.png")
