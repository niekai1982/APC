# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 18:53:08 2017

@author: nieka
"""

import cv2
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def convTri(I, *arg):
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
    J = cv2.filter2D(I, -1, f)
    J = cv2.filter2D(I, -1, f.T)
    if s > 1:
        t = np.int(np.floor(s * 1. / 2) + 1)
        J = J[(t - 1):(J.shape[0]-s+t):s, (t - 1):(J.shape[1]-s+t):s]
    return J
   
    
if __name__ == '__main__':
    r = 1
    s = 2

    data_mat = sio.loadmat('peppers_test.mat')
    img = data_mat['I']

    I = img.astype(np.single)
    J = convTri(I, r, s)
    
    plt.imshow(J)
    plt.show()
