__author__ = 'kai'
# ! -*- coding: UTF-8 -*-

"""
INPUTS
  I           - [hxwx3] input image (uint8 or single/double in [0,1])

"""

import cv2
import numpy as np
import feature.stub as stub
import feature.vector as vector
import misc.gradient as grad
import matplotlib.pyplot as plt
from time import time

GRAD_DDEPTH = cv2.CV_16S  # Word size for gradient channels.
ORIENTATION_DEGREES = [15, 45, 75, 105, 135, 165]  # Central degrees for gradient orientation channels.
ORIENTATION_BIN = 14  # Bin size for orientation channels.

class PChns(object):
    """
    INPUT PARAMETERS
    pChns       - parameters (struct or name/value pairs)
        .shrink       - [4] integer downsampling amount for channels
        .pColor       - parameters for color space:
            .enabled      - [1] if true enable color channels
            .smooth       - [1] radius for image smoothing (using convTri)
            .colorSpace   - ['luv'] choices are: 'gray', 'rgb', 'hsv', 'orig'
        .pGradMag     - parameters for gradient magnitude:
            .enabled      - [1] if true enable gradient magnitude channel
            .colorChn     - [0] if>0 color channel to use for grad computation
            .normRad      - [5] normalization radius for gradient
            .normConst    - [.005] normalization constant for gradient
            .full         - [0] if true compute angles in [0,2*pi) else in [0,pi)
        .pGradHist    - parameters for gradient histograms:
            .enabled      - [1] if true enable gradient histogram channels
            .binSize      - [shrink] spatial bin size (defaults to shrink)
            .nOrients     - [6] number of orientation channels
            .softBin      - [0] if true use "soft" bilinear spatial binning
            .useHog       - [0] if true perform 4-way hog normalization/clipping
            .clipHog      - [.2] value at which to clip hog histogram bins
        .pCustom      - parameters for custom channels (optional struct array):
            .enabled      - [1] if true enable custom channel type
            .name         - ['REQ'] custom channel type name
            .hFunc        - ['REQ'] function handle for computing custom channels
            .pFunc        - [{}] additional params for chns=hFunc(I,pFunc{:})
            .padWith      - [0] how channel should be padded (e.g. 0,'replicate')
        .complete     - [] if true does not check/set default vals in pChns
    """

    def __init__(self):
        self.shrink = 4
        self.pColor = self.pColor()
        self.pGradMag = self.pGradMag()
        self.pGradHist = self.pGradHist(shrink=self.shrink)
        self.pCustom = self.pCustom()
        self.complete = True

    class pColor(object):
        def __init__(self):
            self.enabled = 1
            self.smooth = 1
            self.colorSpace = "luv"

    class pGradMag(object):
        def __init__(self):
            self.enabled = 1
            self.colorChn = 0
            self.normRad = 5
            self.normConst = 0.005
            self.full = 0

    class pGradHist(object):
        def __init__(self, shrink):
            self.enabled = 1
            self.binSize = shrink
            self.nOrients = 6
            self.softBin = 0
            self.useHog = 0
            self.clipHog = 0.2

    class pCustom(object):
        def __init__(self):
            self.enabled = 1
            self.name = 'REQ'
            self.hFunc = 'REQ'
            self.pFunc = []
            self.padWith = 0


class Chns(object):
    """
    OUTPUTS
    chns       - output struct
        .pChns      - exact input parameters used
        .nTypes     - number of channel types
        .data       - [nTypes x 1] cell [h/shrink x w/shrink x nChns] channels
        .info       - [nTypes x 1] struct array
        .name       - channel type name
        .pChn       - exact input parameters for given channel type
        .nChns      - number of channels for given channel type
        .padWith    - how channel should be padded (0,'replicate')
    """

    def __init__(self, pChns, nType=10):
        self.pChns  = pChns
        self.nTypes = 0
        self.data   = []
        self.info   = self.info()

    class info(object):
        def __init__(self):
            self.name    = []
            self.pChn    = []
            self.nChns   = []
            self.padWith = []


class ChnsCompute(object):

    def __init__(self):
        self.pChns = PChns()
        self.chns  = Chns(self.pChns)

    def compute(self, I):
        start = time()
        self.pre_process(I)
        # end = time()
        # print "preprocess spend time : %f" % (end - start)
        self.colorChnCompute()
        # end1 = time()
        # print "colorpreprocess spend time : %f" % (end1 - end)
        self.gradMagChnCompute()
        # end2 = time()
        # print "gradMagpreprocess spend time : %f" % (end2 - end1)
        self.gradHistChnCompute()
        # end3 = time()
        # print "gradHistpreprocess spend time : %f" % (end3 - end2)
        end = time()
        print "chnsCompute spend time : %f" % (end - start)

    def pre_process(self, I):

        self.I = self.crop_img(I, self.pChns.shrink)
        self.h, self.w = self.I.shape[:2]
        self.h /= self.pChns.shrink
        self.w /= self.pChns.shrink

        self.gray_img   = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        self.gray_img   = cv2.GaussianBlur(self.gray_img, (3, 3), 0)

        self.gradient_x = cv2.Sobel(self.gray_img, GRAD_DDEPTH, 1, 0)
        self.gradient_y = cv2.Sobel(self.gray_img, GRAD_DDEPTH, 0, 1)

        self.gx_scaled  = cv2.convertScaleAbs(self.gradient_x)
        self.gy_scaled  = cv2.convertScaleAbs(self.gradient_y)

        self.magnitude  = cv2.addWeighted(self.gx_scaled, 0.5, self.gy_scaled, 0.5, 0)

    def crop_img(self, I, shrink):
        # crop I so divisible by shrink and get target dimensions
        ndim_I = len(I.shape)
        assert ndim_I > 1
        h, w = I.shape[:2]
        cr = np.mod([h, w], shrink)
        if np.any(cr):
            h = h - cr[0]
            w = w - cr[1]
        I = I[:h, :w]
        return I

    def convTri(self, I, r, s):
        pass

    def oriented_gradient(self, grad_x, grad_y, degree, bin_size):
        """
        Returns the oriented gradient channel.

        :param grad_x: Gradient computed only for X axis.
        :param grad_y: Gradient computed only for Y axis.
        :param degree: Degree of the edge to be calculated
        :param bin_size: Degree margin for which the edges to be calculated.

        For example, if degree is '30' and bin size is '10', this routine computes edges for the degree interval 20 to 40.
        """
        assert grad_x.shape == grad_y.shape

        lower_bound = degree - bin_size
        upper_bound = degree + bin_size

        rows, cols = grad_x.shape

        oriented = np.zeros((rows, cols), np.uint8)

        d = np.arctan2(grad_y, grad_x) * 180 / np.pi
        mask = (d > lower_bound) * (d < upper_bound)
        oriented[mask] = 255

        return oriented

    def addChn(self, data, name, pChn, padWith, h, w):
        h1, w1 = data.shape[:2]
        if h1 != h or w1 != w:
            data = cv2.resize(data, (w, h))
            assert np.all(np.mod(np.array([h1, w1], np.float) / np.array([h, w], np.float),1) == 0)
        data = cv2.split(data)
        self.chns.data.append(data)
        self.chns.nTypes += 1
        self.chns.info.name.append(name)
        self.chns.info.pChn.append(pChn)
        self.chns.info.nChns.append(len(data))
        self.chns.info.padWith.append(padWith)

    def colorChnCompute(self):
        """
        compute color channels
        """
        p = self.pChns.pColor
        nm = 'color channels'
        luv = cv2.cvtColor(self.I, cv2.COLOR_BGR2LUV)
        if p.enabled:
            self.addChn(luv, nm, p, 'replicate', self.h, self.w)

    def gradMagChnCompute(self):
        """
        compute color channels
        """
        p = self.pChns.pGradMag
        nm = 'gradient magnitude'

        if  p.enabled:
            self.addChn(self.magnitude, nm, p, 0, self.h, self.w)
            # self.addChn(I, nm, p, 'replicate', h, w)

    def gradHistChnCompute(self):
        assert len(ORIENTATION_DEGREES) == 6
        assert min(ORIENTATION_DEGREES) - ORIENTATION_BIN > 0
        assert max(ORIENTATION_DEGREES) + ORIENTATION_BIN < 180

        p = self.pChns.pGradHist
        nm = 'gradient histogram'

        for i, deg in enumerate(ORIENTATION_DEGREES):
            orie = self.oriented_gradient(self.gradient_x, self.gradient_y, deg, ORIENTATION_BIN)
            orie = cv2.medianBlur(orie, 3)
            orie = cv2.bitwise_and(orie, self.magnitude)
            self.H = orie.copy() if not i else np.dstack((self.H, orie))
        if p.enabled:
            self.addChn(self.H, nm, p, 0, self.h, self.w)


if __name__ == "__main__":
    img = cv2.imread('test.jpg')

    start_t = time()
    chn_cp = ChnsCompute()
    chn_cp.compute(img)
    end_t = time()
    print chn_cp.chns.nTypes

    # print chn_cp.chns.info.name
    # print chn_cp.chns.info.nChns
    # print chn_cp.chns.info.pChn
    # print chn_cp.chns.info.padWith

    print "total spend time : %f" % (end_t - start_t)

    for i in range(len(chn_cp.chns.data)):
        for j in range(len(chn_cp.chns.data[i])):
            plt.imshow(chn_cp.chns.data[i][j])
            plt.colorbar()
            plt.show()
