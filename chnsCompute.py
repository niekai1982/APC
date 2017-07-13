__author__ = 'kai'
# ! -*- coding: UTF-8 -*-

"""
INPUTS
  I           - [hxwx3] input image (uint8 or single/double in [0,1])

"""

import cv2
import numpy as np
from imgProc import convTri
from getFeature_test import rgb2luv, gradient_Mag, gradient_Hist, rgb2luv_test
from time import time


# chns struct
class C_pChns(object):
    def __init__(self, **data):
        self.__dict__.update(data)


class C_Color(object):
    def __init__(self, **data):
        self.__dict__.update(data)


class C_GradMag(object):
    def __init__(self, **data):
        self.__dict__.update(data)


class C_GradHist(object):
    def __init__(self, **data):
        self.__dict__.update(data)


class C_Custom(object):
    def __init__(self, **data):
        self.__dict__.update(data)


# output struct
class C_info(object):
    def __init__(self, **data):
        self.__dict__.update(data)


class C_chns(object):
    def __init__(self, **data):
        self.__dict__.update(data)


def chnsCompute(I=[], *varargin):
    # get default parameters pChns
    if varargin:
        pChns = varargin
    else:
        pChns = C_pChns()
        pChns.shrink = 4
        pChns.pColor = C_Color(enabled=1, smooth=1, colorSpace='luv')
        pChns.pGradMag = C_GradMag(enabled=1, colorChn=0, normRad=5, normConst=.005, full=0)
        pChns.pGradHist = C_GradHist(enable=1, binSize=[], nOrients=6, softBin=0, useHog=0, clipHog=.2)
        pChns.pCustom = C_Custom(enable=1, name='REQ', hFunc='REQ', pFunc={}, padWith=0)
        pChns.complete = 1
    if I is []:
        chns = pChns
        return chns

    # create output struct
    info = C_info(name=[], pChn=[], nChns=[], padWidth=[])
    chns = C_chns(pChns=pChns, nTypes=0, data=[], info=info)

    # crop I so divisible by shrink and get target dimensions
    shrink = pChns.shrink
    h, w = I.shape[:2]
    cr = np.mod([h, w], shrink)
    if np.any(cr):
        h = h - cr[0]
        w = w - cr[1]
    I = I[:h, :w]

    h /= shrink
    w /= shrink

    # compute color channels
    start = time()
    p = pChns.pColor
    nm = 'color channels'
    I = rgb2luv_test(I, 1./255)
    _,I = convTri(I, p.smooth, 1)
    addChn(chns, I, nm, p, 0, h, w)
    end = time()
    print end - start

    # compute gradMag channels
    start = time()
    p = pChns.pGradMag
    nm = 'gradient magnitude'
    if pChns.pGradHist.enable:
        M, O = gradient_Mag(I, pChns.pGradMag.normRad, pChns.pGradMag.normConst, cv_flag=1)
    addChn(chns, M, nm, p, 0, h, w)
    end = time()
    print end - start

    # compute gradent histogram channels
    start = time()
    p = pChns.pGradHist
    nm = 'gradient histogram'
    if pChns.pGradHist.enable:
        binSize = p.binSize
        if not binSize:
            binSize = pChns.shrink
    H = gradient_Hist(M, O, binSize, p.nOrients, p.softBin, 0)
    addChn(chns, H, nm, p, 0, h, w)
    end = time()
    print end - start

    return chns


def addChn(chns, data, name, pChn, padWith, h, w):
    '''
    help function to add a channel to chns.
    '''
    h1, w1 = data.shape[:2]
    if h1 != h or w1 != w:
        # imResampleMex differents
        data = cv2.resize(data, (w, h))
        assert np.all(np.mod(np.array([h1, w1], np.float) / np.array([h, w], np.float), 1) == 0)
    data = cv2.split(data)
    chns.data.append(data)
    chns.nTypes += 1
    chns.info.name.append(name)
    chns.info.pChn.append(pChn)
    chns.info.nChns.append(len(data))
    chns.info.padWidth.append(padWith)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time

    I = cv2.imread('hiv00000_06960.jpg')
    # I = (I * 1. / 255).astype(np.float32)
    start = time()
    chns = chnsCompute(I)
    end = time()
    print (end - start)

    # for i in range(chns.nTypes):
    #     for j in range(chns.info.nChns[i]):
    #         plt.imshow(chns.data[i][j])
    #         plt.show()
