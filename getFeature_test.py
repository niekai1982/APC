import cv2
import numpy as np
from imgProc import convTri
from time import time

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


def gradient(img, cv_flag):
    h, w = img.shape[:2]
    if cv_flag:
        Gx = cv2.filter2D(img, -1, kernel_x)
        Gy = cv2.filter2D(img, -1, kernel_y)

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[h - 1, :] = img[h - 1, :] - img[h - 2, :]
        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, w - 1] = img[:, w - 1] - img[:, w - 2]
    else:
        Gy, Gx = np.gradient(img, 1, axis=(0, 1))
    return Gx, Gy


def gradient_Mag(img, normRad, normConst, cv_flag):
    if (len(img.shape) > 2):
        c = img.shape[-1]
    else:
        c = 1
    if cv_flag:
        Gx, Gy = gradient(img, cv_flag)
        M, O = cv2.cartToPolar(Gx, Gy, angleInDegrees=False)
        O[O > np.pi] -= np.pi
    else:
        Gx, Gy = gradient(img, cv_flag)
        M = np.sqrt(Gx ** 2 + Gy ** 2)
        O = np.arctan2(Gy, Gx)
    if c > 1:
        max_idx = np.argmax(M, axis=2)
        M_out = np.zeros(img.shape[:2])
        O_out = np.zeros(img.shape[:2])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                M_out[i, j] = M[i, j, max_idx[i, j]]
                O_out[i, j] = O[i, j, max_idx[i, j]]
        _,S = convTri(M_out, normRad)
        M_out = M_out / (S + normConst)
        return  M_out, O_out
    else:
        M = convTri(M, normRad)
        M = M / (M + normConst)
        return M, O


def gradient_Hist(M, O, bin, nOrients, softBin, full):
    h, w = M.shape
    nb = w * h
    s = 1. * bin
    sInv2 = 1 / s / s
    oMult = nOrients / (2 * np.pi) if full else nOrients / np.pi
    oMax = nOrients * nb

    o = O * oMult
    o0 = o.astype(dtype=np.int)
    od = o - o0
    o0 *= nb
    o0[o0 >= oMax] = 0
    O0 = o0.copy()

    o1 = o0 + nb
    o1[o1 == oMax] = 0
    O1 = o1.copy()
    m = M * sInv2
    M1 = od * m
    M0 = m - M1

    O0 = O0.flatten()
    O1 = O1.flatten()
    M0 = M0.flatten()
    M1 = M1.flatten()

    ord_t = np.arange(w * h)

    O0 += ord_t
    O1 += ord_t

    H = np.zeros(h * w * nOrients, dtype=np.float64)

    for elem in range(O0.shape[0]):
        H[O0[elem]] += M0[elem]
        H[O1[elem]] += M1[elem]

    H.shape = nOrients, h, w

    H1 = np.zeros((h / bin, w / bin, nOrients))

    for i in range(h / bin):
        for j in range(w / bin):
            for k in range(nOrients):
                H1[i, j, k] = H[k, i * bin:(i + 1) * bin, j * bin:(j + 1) * bin].sum()
    return H1


if __name__ == '__main__':
    import os
    import scipy.io as sio
    import profile
    import matplotlib.pyplot as plt
    from time import time

    img = cv2.imread('peppers.png')
    luv = rgb2luv(img, 1.0 / 255)
