__author__ = 'kai'

from chnsCompute import *

eps = 2.2204e-16


class pPyramid(object):
    """
    INPUTS
      I            - [hxwx3] input image (uint8 or single/double in [0,1])
      pPyramid     - parameters (struct or name/value pairs)
       .pChns        - parameters for creating channels (see chnsCompute.m)
       .nPerOct      - [8] number of scales per octave
       .nOctUp       - [0] number of upsampled octaves to compute
       .nApprox      - [-1] number of approx. scales (if -1 nApprox=nPerOct1)
       .lambdas      - [] coefficients for power law scaling (see BMVC10)
       .pad          - [0 0] amount to pad channels (along T/B and L/R)
       .minDs        - [16 16] minimum image size for channel computation
       .smooth       - [1] radius for channel smoothing (using convTri)
       .concat       - [1] if true concatenate channels
       .complete     - [] if true does not check/set default vals in pPyramid
    """

    def __init__(self):
        self.nPerOct = 8
        self.nOctUp = 0
        self.nApprox = -1
        self.lambdas = []
        self.pad = [0, 0]
        self.minDs = [16, 16]
        self.smooth = 1
        self.concat = 1
        self.complete = True


class pyramid(object):
    """
    OUTPUTS
    pyramid      - output struct
    .pPyramid     - exact input parameters used (may change from input)
    .nTypes       - number of channel types
    .nScales      - number of scales computed
    .data         - [nScales x nTypes] cell array of computed channels
    .info         - [nTypes x 1] struct array (mirrored from chnsCompute)
    .lambdas      - [nTypes x 1] scaling coefficients actually used
    .scales       - [nScales x 1] relative scales (approximate)
    .scaleshw     - [nScales x 2] exact scales for resampling h and w
    """

    def __init__(self, pPyramid, nTypes=10, nScales=1):
        self.pPyramid = pPyramid
        self.nTypes = nTypes
        self.nScales = nScales
        self.data = []
        self.info = []
        self.lambdas = []
        self.scales = []
        self.scaleshw = []


class ChnsPyramid(object):
    def __init__(self):
        self.pPyramid = pPyramid()
        self.pyramid = pyramid(self.pPyramid)
        self.chns = ChnsCompute()
        self.nApprox = 7
        self.scales = []
        self.scaleshw = []
        self.nScales = 0
        self.isA = []
        self.isR = []
        self.data = []


    def getScales(self, sz):
        if 0 in sz:
            self.scales = []
            self.scaleshw = []
            return
        self.nScales = np.floor(self.pPyramid.nPerOct * (
            self.pPyramid.nOctUp + np.log2(np.min(np.true_divide(sz, self.pPyramid.minDs)))) + 1)
        self.scales = np.exp2(-np.arange(self.nScales) / self.pPyramid.nPerOct + self.pPyramid.nOctUp)
        d = sz[::1] if sz[0] < sz[1] else sz[::-1]
        for i in range(int(self.nScales)):
            s = self.scales[i]
            s0 = (np.round(d[0] * s / self.chns.pChns.shrink) * self.chns.pChns.shrink - .25 * self.chns.pChns.shrink) / \
                 d[0]
            s1 = (np.round(d[0] * s / self.chns.pChns.shrink) * self.chns.pChns.shrink + .25 * self.chns.pChns.shrink) / \
                 d[0]
            ss = np.arange(0, 1 + eps, 0.01) * (s1 - s0) + s0
            es0 = d[0] * ss
            es0 = np.abs(es0 - np.round(es0 / self.chns.pChns.shrink) * self.chns.pChns.shrink)
            es1 = d[1] * ss
            es1 = np.abs(es1 - np.round(es1 / self.chns.pChns.shrink) * self.chns.pChns.shrink)
            x = np.argmin(np.maximum(es0, es1))
            self.scales[i] = ss[x]
        kp = self.scales[:-1] != self.scales[1:]
        kp = np.hstack((kp, True))
        self.scales = self.scales[kp]
        self.scaleshw = np.round(sz[0] * self.scales / self.chns.pChns.shrink) * self.chns.pChns.shrink / sz[0]
        self.scaleshw = np.stack(
            (self.scaleshw, np.round(sz[1] * self.scales / self.chns.pChns.shrink) * self.chns.pChns.shrink / sz[1]),
            axis=-1)
        self.nScales = self.scales.shape[0]
        self.isR = np.arange(0, self.nScales, self.nApprox + 1)
        self.isA = np.arange(self.nScales)
        self.isA = np.delete(self.isA, self.isR)
        self.data = [None] * self.nScales


    def chnsPyramidCompute(self, I):
        sz = I.shape[:2]
        self.getScales(sz)
        for i in self.isR:
            scale = self.scales[i]
            sz1 = np.round(np.true_divide(np.multiply(sz, scale), self.chns.pChns.shrink)) * self.chns.pChns.shrink
            if np.all(sz == sz1):
                I1 = I
            else:
                I1 = cv2.resize(I, (int(sz1[1]), int(sz1[0])))
            if (scale == .5 and self.pPyramid.nApprox > 0 or self.pPyramid.nPerOct == 1):
                I = I1
            self.chns.compute(I1)


if __name__ == '__main__':
    start = time()
    I = cv2.imread('test.jpg')
    test_input = ChnsPyramid()
    test_input.pPyramid.minDs = [128, 128]
    test_input.chnsPyramidCompute(I)
    end = time()
    print "total spend time : %f" % (end - start)

