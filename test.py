import os
import cv2
import pdb
import time
import argparse

import imutils
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian
from BrainTest import Get_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from common import draw_str
from chnsCompute import ChnsCompute


def forestInds(data, thrs, fids, child, N):
    # could be optimize
    inds = np.zeros((N, 1), dtype=np.int)
    out = np.ones((N, 1), dtype=np.int)
    # for i in range(N):
    #     k = 0
    #     while child[k]:
    #             k = child[k] - 1
    #         else:
    #             k = child[k]
    #     inds[i] = k
    out[data[:, fids[0]] > thrs[0]] = 2
    return out


def pyramid(image, scale=1.2, minSize=(256, 128)):
    yield scale, image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield scale, image


def sliding_window(imageSize, stepSize, windowSize):
    for y in xrange(0, imageSize[0] - windowSize[1] + 1, stepSize):
        for x in xrange(0, imageSize[1] - windowSize[0] + 1, stepSize):
            yield (x, y)


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


    path = r'E:\PROGRAM\APC\sample_test\3'
    files = os.listdir(path)
    stop_flag = False
    continue_flag = True
    for file in files:
        
        image = cv2.imread(os.path.join(path, file))
        image = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
        vis = image.copy()
        
        start = time.time()
        chn_cp = ChnsCompute()
        chn_cp.compute(image)
       
        feat = []
        coor_x = []
        coor_y = []
        
        
        
        for (x, y) in sliding_window(image.shape, 8, (64, 32)):
            out = np.hstack([chn_cp.chns.data[i][j][y/4:(y/4 + 8), x/4:(x/4 + 16)].flatten() for i in range(len(chn_cp.chns.data)) for j in range(len(chn_cp.chns.data[i]))])
            feat.append(out)
            coor_x.append(x)
            coor_y.append(y)

        feat = np.array(feat)
        
        end = time.time()
        print (end - start)
        
        nWin, nWinFeat = feat.shape
        hs_out = np.zeros((nWin, 1))
        
        for i in range(nWeaks):
            ids = forestInds(feat, thrs[:,i], fids[:,i], child[:,i], nWin)
            hs_out = hs_out + hs[ids, i]
            
        
        
        coor_x = np.array(coor_x)
        coor_y = np.array(coor_y)
        
        coor_x.shape = -1,1
        coor_y.shape = -1,1

        test_x = coor_x[hs_out > 15]
        test_y = coor_y[hs_out > 15]

        for (x, y) in zip(test_x, test_y):
            cv2.rectangle(vis, (x, y), (x + 64, y + 32), (0,0,255),1)
        end = time.time()
       

        cv2.imshow("window", vis)
        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.destroyAllWindows()
            break
#        while True:
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
