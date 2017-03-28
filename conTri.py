# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 18:53:08 2017

@author: nieka
"""

import cv2
import scipy.io as sc_io
import skimage.io as sk_io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def convTri(I, *arg):
    h, w, n_dim = I.shape
    J = np.empty_like(I)
    if len(arg) < 2:
        s = 1
        r = arg[0]
    else:
        r = arg[0]
        s = arg[1]
    if r<=1:
        p = 12 / r / (r + 2) - 2
        f = np.array([1, p, 1]) * 1. / (2 + p)
        r = 1
    else:
        f = range(1, r + 1) + [r + 1] + range(1, r + 1)[::-1]
        f = np.array(f, dtype=np.single) / (r + 1) ** 2
    f.shape = 1, -1
    for nd in range(n_dim):
        I_d = np.pad(I[:,:,nd], r, mode='symmetric')
        J1 = signal.fftconvolve(I_d, f.T, mode='valid')
        J2 = signal.fftconvolve(J1, f, mode='valid')
        J[:,:,nd] = J2
    if s > 1:
        t = np.int(np.floor(s * 1. / 2) + 1)
        J = J[(t - 1):(J.shape[0]-s+t):s, (t - 1):(J.shape[1]-s+t):s]
    return J


if __name__ == '__main__':
    r = 1
    s = 2

    img = sk_io.imread('peppers.png')

    I = img.astype(np.single)
    I = I / 255
    J = convTri(I, r, s)
    
    plt.imshow(J)
    plt.show()
