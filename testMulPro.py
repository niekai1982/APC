# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:12:20 2017

@author: nieka
"""

import cv2
import numpy as np
from time import time
from multiprocessing import Process
from multiprocessing import Pool


kernel_x = np.array([[0, 0, 0], [-1. / 2, 0, 1. / 2], [0, 0, 0]])
kernel_y = np.array([[0, -1. / 2, 0], [0, 0, 0], [0, 1. / 2, 0]])

def rgb2luv_setup(nrm):
    z = nrm
    y0 = (6.0 / 29) * (6.0 / 29) * (6.0 / 29)
    a = (29.0 / 3) * (29.0 / 3) * (29.0 / 3)
    un = 0.197833
    vn = 0.468331

    mr = np.zeros((3, 1), dtype=np.float64)
    mg = np.zeros((3, 1), dtype=np.float64)
    mb = np.zeros((3, 1), dtype=np.float64)

    mr[0] = 0.430574 * z
    mr[1] = 0.222015 * z
    mr[2] = 0.020183 * z
    mg[0] = 0.341550 * z
    mg[1] = 0.706655 * z
    mg[2] = 0.129553 * z
    mb[0] = 0.178325 * z
    mb[1] = 0.071330 * z
    mb[2] = 0.939180 * z

    maxi = 1.0 / 270
    minu = -88 * maxi
    minv = -134 * maxi

    lTable = np.zeros((1064, 1), dtype=np.float64)
    lInit = False

    if (lInit):
        return lTable, mr, mg, mb, minu, minv, un, vn

    for i in range(1025):
        y = (i / 1024.0)
        l = 116 * pow(y, 1.0 / 3.0) - 16 if y > y0 else y * a
        lTable[i] = l * maxi

    for i in range(1025, 1064):
        lTable[i] = lTable[i - 1]

    lInit = True
    return lTable, mr, mg, mb, minu, minv, un, vn

def rgb2luv(I, nrm):
    lTable, mr, mg, mb, minu, minv, un, vn = rgb2luv_setup(nrm)
    B, G, R = np.dsplit(I, 3)
    L = np.zeros_like(B, dtype=np.float64)
    U = np.zeros_like(G, dtype=np.float64)
    V = np.zeros_like(R, dtype=np.float64)

    # x = mr[0] * R + mg[0] * G + mb[0] * B
    # y = mr[1] * R + mg[1] * G + mb[1] * B
    # z = mr[2] * R + mg[2] * G + mb[2] * B
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            r = R[i, j, 0]
            g = G[i, j, 0]
            b = B[i, j, 0]
            x = mr[0] * r + mg[0] * g + mb[0] * b
            y = mr[1] * r + mg[1] * g + mb[1] * b
            z = mr[2] * r + mg[2] * g + mb[2] * b
            l = lTable[(int)(y * 1024)]
            L[i, j, 0] = l
            z = 1 / (x + 15 * y + 3 * z + 1e-35)
            U[i, j, 0] = l * (13 * 4 * x * z - 13 * un) - minu
            V[i, j, 0] = l * (13 * 9 * y * z - 13 * vn) - minv
    return np.dstack((L, U, V))

def rgb2luv_test(I, nrm):
    lTable, mr, mg, mb, minu, minv, un, vn = rgb2luv_setup(nrm)
    B, G, R = np.dsplit(I, 3)
    L = np.zeros_like(B, dtype=np.float64)
    U = np.zeros_like(G, dtype=np.float64)
    V = np.zeros_like(R, dtype=np.float64)

    X = mr[0] * R + mg[0] * G + mb[0] * B
    Y = mr[1] * R + mg[1] * G + mb[1] * B
    Z = mr[2] * R + mg[2] * G + mb[2] * B
    Z = 1 / (X + 15 * Y + 3 * Z + 1e-35)
    L = lTable[(Y * 1024).astype(np.int)]
    L.shape = L.shape[0], L.shape[1], L.shape[2]
    U = L * (13 * 4 * X * Z - 13 * un) - minu
    V = L * (13 * 9 * Y * Z - 13 * vn) - minv
    return np.dstack((L, U, V))


def gradient(img, cv_flag):
    h, w = img.shape[:2]
    # print h
    #print h
#   cv2.filter2D(img, -1, kernel_x))
    for i in range(1000):
        cv2.filter2D(img, -1, kernel_y)
        
#        Gy[0, :] = img[1, :] - img[0, :]
#        Gy[h - 1, :] = img[h - 1, :] - img[h - 2, :]
#        Gx[:, 0] = img[:, 1] - img[:, 0]
#        Gx[:, w - 1] = img[:, w - 1] - img[:, w - 2]
#    else:
#        Gy, Gx = np.gradient(img, 1, axis=(0, 1))
#    return Gx, Gy

if __name__ == '__main__':
    img = cv2.imread('hiv00000_06960.jpg')
    img = cv2.resize(img, (200, 100))
    start = time()
    out = rgb2luv(img, 1./255)
    end = time()
    print end - start
    start = time()
    out1 = rgb2luv_test(img, 1./255)
    end = time()
    print end - start