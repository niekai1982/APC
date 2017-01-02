import os
import time

import cv2
import imutils
import numpy as np
import scipy.io as sio
from STC import STC
import matplotlib.pyplot as plt

from chnsCompute import ChnsCompute

GRAD_DDEPTH = cv2.CV_16S  # Word size for gradient channels.
ORIENTATION_DEGREES = [15, 45, 75, 105, 135, 165]  # Central degrees for gradient orientation channels.
ORIENTATION_BIN = 14  # Bin size for orientation channels.



def oriented_gradient(degree_mat, degree, bin_size):
    """
    Returns the oriented gradient channel.

    :param grad_x: Gradient computed only for X axis.
    :param grad_y: Gradient computed only for Y axis.
    :param degree: Degree of the edge to be calculated
    :param bin_size: Degree margin for which the edges to be calculated.

    For example, if degree is '30' and bin size is '10', this routine computes edges for the degree interval 20 to 40.
    """

    lower_bound = degree - bin_size
    upper_bound = degree + bin_size

    rows, cols = degree_mat.shape

    oriented = np.zeros((rows, cols), np.uint8)

    mask = (degree_mat > lower_bound) * (degree_mat < upper_bound)
    oriented[mask] = 255

    return oriented



def crop_img(I, shrink):
    ndim_I = len(I.shape)
    assert ndim_I > 1
    h, w = I.shape[:2]
    cr = np.mod([h, w], shrink)
    if np.any(cr):
        h = h - cr[0]
        w = w - cr[1]
    I = I[:h, :w]
    return I


def get_feature(I, shrink):
    I = crop_img(I, shrink)
    h, w = I.shape[:2]
    h /= shrink
    w /= shrink

    gray_img = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    gradient_x = cv2.Sobel(gray_img, GRAD_DDEPTH, 1, 0)
    gradient_y = cv2.Sobel(gray_img, GRAD_DDEPTH, 0, 1)

    gx_scaled = cv2.convertScaleAbs(gradient_x)
    gy_scaled = cv2.convertScaleAbs(gradient_y)

    magnitude = cv2.addWeighted(gx_scaled, 0.5, gy_scaled, 0.5, 0)

    luv = cv2.cvtColor(I, cv2.COLOR_BGR2LUV)

    assert len(ORIENTATION_DEGREES) == 6
    assert min(ORIENTATION_DEGREES) - ORIENTATION_BIN > 0
    assert max(ORIENTATION_DEGREES) + ORIENTATION_BIN < 180

    degree = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    for i, deg in enumerate(ORIENTATION_DEGREES):
        orie = oriented_gradient(degree, deg, ORIENTATION_BIN)
        orie = cv2.medianBlur(orie, 3)
        orie = cv2.bitwise_and(orie, magnitude)
        H = orie.copy() if not i else np.dstack((H, orie))

    luv = cv2.resize(luv, (w, h))
    magnitude = cv2.resize(magnitude, (w, h))
    H = cv2.resize(H, (w, h))

    return luv, magnitude, H


def forestInds(data, thrs, fids, child, N, nfeats):
    # could be optimize
    nFeats_perChn = data.shape[-1] / 10
    chn_idx = fids[0] / nFeats_perChn
    feat_idx = fids[0] % nFeats_perChn

    inds = np.zeros((N, 1), dtype=np.int)
    out = np.ones((N, 1), dtype=np.int)

    idx = feat_idx * 10 + chn_idx
    # for i in range(N):
    #     k = 0
    #     while child[k]:
    #             k = child[k] - 1
    #         else:
    #             k = child[k]
    #     inds[i] = k
    out[data[:, idx] > thrs[0]] = 2
    return out


def pyramid(image, scale=1.2, minSize=(256, 128)):
    yield scale, image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield scale, image


def sliding_window(imageSize, stepSize, windowSize, chns_feat):
    for y in xrange(0, imageSize[0] - windowSize[1] + 1, stepSize):
        for x in xrange(0, imageSize[1] - windowSize[0] + 1, stepSize):
            # feat_veator = np.hstack([chns_feat[y / 4 : (y / 4 + 8), x / 4 : (x / 4 + 16), i].flatten() for i in range(chns_feat.shape[-1])])
            feat_vector = chns_feat[y/4:y/4 + 8,x/4:x/4+16].flatten()
            yield (x, y, feat_vector)


def sliding_window_test(imageSize, stepSize, windowSize, chns_feat):
    out = []
    for y in xrange(0, imageSize[0] - windowSize[1] + 1, stepSize):
        for x in xrange(0, imageSize[1] - windowSize[0] + 1, stepSize):
            # feat_veator = np.hstack([chns_feat[y / 4 : (y / 4 + 8), x / 4 : (x / 4 + 16), i].flatten() for i in range(chns_feat.shape[-1])])
            chns_feat_vector = chns_feat[y/4:y/4 + 8,x/4:x/4+16].flatten()
            out.append(chns_feat_vector)
    return out


if __name__ == '__main__':

    test = sio.loadmat('model.mat')
    fids = test['model'][0][0][0]
    thrs = test['model'][0][0][1]
    child = test['model'][0][0][2]
    hs = test['model'][0][0][3]
    weights = test['model'][0][0][4]
    depth = test['model'][0][0][5]
    errs = test['model'][0][0][6]
    losses = test['model'][0][0][7]
    treeDepth = test['model'][0][0][8]

    nWeaks = fids.shape[1]

    path = r'E:\PROGRAM\APC\sample_test\1'
    files = os.listdir(path)
    stop_flag = False
    continue_flag = True
    tracker_list = []

    for file in files:

        image = cv2.imread(os.path.join(path, file))
        image = cv2.resize(image, (image.shape[1] / 2, image.shape[0] / 2))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis = image.copy()


        start = time.time()
        tracker_temp = []
        for tracker in tracker_list:
            pos_p = tracker.pos
            tracker.updata_tracker(gray)
            pos_n = tracker.pos
            if abs(pos_p[0] - pos_n[0]) > 1:
                tracker.draw_state(vis)
                tracker_temp.append(tracker)
        tracker_list = tracker_temp

        luv, m, h = get_feature(image, shrink=4)
        chns_feat = np.dstack((luv,m,h))


        # print "feature extract spend time : %f" % (end - start)
        #
        # start = time.time()
        # chn_cp = ChnsCompute()
        # chn_cp.compute(image)
        #
        feat_mat = []
        coor_x = []
        coor_y = []
        #
        # out = sliding_window_test(image.shape,8,(64,32),chns_feat)
        for (x, y, feat_vector) in sliding_window(image.shape, 8, (64, 32), chns_feat):
            feat_mat.append(feat_vector)
            coor_x.append(x)
            coor_y.append(y)

        end1 = time.time()
        # print "test time spend : %f" % (end1 - end)
        #
        feat_mat = np.array(feat_mat)
        #
        #
        nWin, nWinFeat = feat_mat.shape
        hs_out = np.zeros((nWin, 1))

        for i in range(nWeaks):
            ids = forestInds(feat_mat, thrs[:, i], fids[:, i], child[:, i], nWin, nWinFeat)
            hs_out = hs_out + hs[ids, i]

        coor_x = np.array(coor_x)
        coor_y = np.array(coor_y)

        coor_x.shape = -1, 1
        coor_y.shape = -1, 1

        test_x = coor_x[hs_out > 16]
        test_y = coor_y[hs_out > 16]


        for (x, y) in zip(test_x, test_y):
            tracker = STC(gray, [x, y, x + 64, y + 32])
            tracker_list.append(tracker)
            cv2.rectangle(vis, (x, y), (x + 64, y + 32), (0, 255, 255), 1)


        end_total = time.time()
        print "total spend time : %f" % (end_total - start)
        cv2.namedWindow("window", 0)
        cv2.imshow("window", vis)
        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.destroyAllWindows()
            break
# while True:
#            ch = cv2.waitKey(1)
#            if ch == 27:
#                stop_flag = True
#                break
#            if ch == ord(' '):
#                continue_flag = True
#                break
#        if stop_flag:
#            cv2.destroyAllWindows()
#            stop_flag = False
#            break
#        if continue_flag:
#            continue_flag = False
#            continue
