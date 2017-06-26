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
    if len(I.shape) >2:
        h, w, n_dim = I.shape
    else:
        h, w = I.shape
        n_dim = 1
    J = np.empty_like(I)
    J_cv = np.empty_like(I)
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
    if n_dim > 1:
        for nd in range(n_dim):
            I_d = np.pad(I[:,:,nd], r, mode='symmetric')

            J1 = signal.fftconvolve(I_d, f.T, mode='valid')
            J2 = signal.fftconvolve(J1, f, mode='valid')
            J[:,:,nd] = J2

            I_d_cv = cv2.copyMakeBorder(I[:,:,nd], r,r,r,r,cv2.BORDER_REFLECT)
            J1_cv = cv2.filter2D(I_d_cv, -1, f)
            J2_cv = cv2.filter2D(J1_cv, -1, f.T)
            J_cv[:,:,nd] = J2_cv[r:-r,r:-r]

    else:
        I_d = np.pad(I, r, mode='symmetric')
        J1 = signal.fftconvolve(I_d, f.T, mode='valid')
        J2 = signal.fftconvolve(J1, f, mode='valid')
        J = J2

        I_d_cv = cv2.copyMakeBorder(I, r,r,r,r,cv2.BORDER_REFLECT)
        J1_cv = cv2.filter2D(I_d_cv, -1, f)
        J2_cv = cv2.filter2D(J1_cv, -1, f.T)
        J_cv = J2_cv[r:-r, r:-r]

    if s > 1:
        t = np.int(np.floor(s * 1. / 2) + 1)
        J = J[(t - 1):(J.shape[0]-s+t):s, (t - 1):(J.shape[1]-s+t):s]
        J_cv = J_cv[(t - 1):(J_cv.shape[0]-s+t):s, (t - 1):(J_cv.shape[1]-s+t):s]
    return J, J_cv


if __name__ == '__main__':
    r = 4
    s = 3


    img = sk_io.imread('peppers.png')

    I = img.astype(np.single)
    I = I / 255
    J, J_cv = convTri(I, r, s)


    plt.imshow(J_cv)
    plt.colorbar()
    plt.show()

    print np.abs(J_cv - J).max()
