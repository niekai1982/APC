import cv2
import numpy as np
from time import time

kernel_x = np.array([[0, 0, 0], [-1. / 2, 0, 1. / 2], [0, 0, 0]])
kernel_y = np.array([[0, -1. / 2, 0], [0, 0, 0], [0, 1. / 2, 0]])

def gradient(img, cv_flag):
    h, w = img.shape[:2]
    if cv_flag:
        Gx = cv2.filter2D(img, -1, kernel_x)
        Gy = cv2.filter2D(img, -1, kernel_y)

        Gy[0, :] =img[1, :] - img[0, :]
        Gy[h - 1, :] = img[h - 1, :] - img[h - 2, :]
        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, w - 1] = img[:, w - 1] - img[:, w - 2]
    else:
        Gy, Gx = np.gradient(img, 1, axis=(0, 1))
    return Gx, Gy


def gradient_Mag(img, cv_flag):
    if (len(img.shape) > 2):
        c = img.shape[-1]
    else:
        c = 1
    if  cv_flag:
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
                M_out[i, j] = M[i, j, max_idx[i,j]]
                O_out[i,j] = O[i,j,max_idx[i,j]]
        return M_out, O_out
    else:
        return M, O


def gradient_Hist(M, O, bin, nOrients, softBin, full):
    h, w = M.shape
    nb = w * h
    s = 1. * bin
    sInv2 = 1 / s / s
    oMult = nOrients / (2 * np.pi) if full else nOrients / np.pi
    oMax  = nOrients * nb

    o  = O * oMult
    o0 = o.astype(dtype=np.int)
    od = o - o0
    o0 *= nb
    o0[o0 >= oMax] = 0
    O0 = o0.copy()

    o1 = o0 + nb
    o1[o1 == oMax] = 0
    O1 = o1.copy()
    m  = M * sInv2
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

    H1 = np.zeros((h/bin, w/bin, nOrients))

    for i in range(h/bin):
        for j in range(w/bin):
            for k in range(nOrients):
                H1[i,j,k] = H[k, i*bin:(i+1)*bin, j*bin:(j+1)*bin].sum()
    return H1


if __name__ == '__main__':
    import os
    import scipy.io as sio
    import profile
    import matplotlib.pyplot as plt
    # work_path = 'C:/Users/nieka/Desktop/test'
    # os.chdir(work_path)

    img = cv2.imread('peppers.png')
    img = img * 1. / 255
    M, O = gradient_Mag(img, 1)
    H = gradient_Hist(M, O, bin=4, nOrients=6, softBin=0, full=1)

    for i in range(6):
        plt.imshow(H[:,:,i])
        plt.show()
    # gray = sio.loadmat('pepers_gray_base.mat')['I']
    #
    # h, w = gray.shape[:2]
    #
    # M_np, O_np = gradient_Mag(gray, cv_flag=0)
    # M_cv, O_cv = gradient_Mag(gray, cv_flag=1)
    #
    # M_m = sio.loadmat('pepers_gray_M')['M']
    # O_m = sio.loadmat('pepers_gray_O')['O']
    # H_m = sio.loadmat('pepers_gray_hist')['H1']
    #
    # H = profile.run("gradient_Hist(M_cv, O_cv, h, w, bin=2, nOrients=6, softBin=0, full=0)")
    # print "end----->>"
    # print "spend time is = %f" % (end - start)
