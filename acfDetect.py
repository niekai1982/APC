import numpy as np
from adaBoostTrain import pModel
from binaryTreeTrain import pData
from chnsCompute import chnsCompute
from chnsCompute import C_chns
from adaBoostApply import adaBoostApply


def getChild(data, fids, thrs, k):
    ftr = data[fids]
    k = 1 if ftr < thrs else 2
    k = k * 2
    return k


def acfDetect(data, clf, shrink=4, modelDsPad=(64, 64), stride=4, cascThr=0.001):
    nTrees = 128
    fids = clf.fids
    thrs = clf.thrs
    hs = clf.hs

    h = 0
    for idx_t in range(nTrees):
        k =  0
        k = getChild(data, fids[k, idx_t], thrs[k, idx_t], k)
        k = getChild(data, fids[k, idx_t], thrs[k, idx_t], k)
        h += hs[k, idx_t]
    return h


if __name__ == '__main__':
    import cPickle
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import cv2
    from time import time
    from common import mosaic


    model = pModel()
    chns = C_chns()

    with open('model.pkl', 'rb') as fp:
        model = cPickle.load(fp)
        fp.close()

    modelDS = [256, 256] # h, w
    step = [8, 8] # h, w
    shrink = 4

    img = cv2.imread('hiv00000_06960.jpg')
    img = img[:,:,::-1]
    h, w = img.shape[:2]

    scale = 0.5
    w = int(w * scale)
    h = int(h * scale)
    img = cv2.resize(img, (w, h))

    start = time()
    chns = chnsCompute(img)
    end = time()
    print "chns feature compute spend time:%f" % (end - start)

    # with open('chns_res_06960', 'wb') as fp:
    #     cPickle.dump(chns, fp)
    #     fp.close()

    # with open('chns_res_06960', 'rb') as fp:
    #     chns = cPickle.load(fp)
    #     fp.close()

    matFtr = []
    for i in range(len(chns.data)):
        for j in range(len(chns.data[i])):
            matFtr.append(chns.data[i][j])
    matFtr = np.array(matFtr)

    xv = np.arange(0, w - 1 - modelDS[1], step[1])
    yv = np.arange(0, h - 1 - modelDS[0], step[0])

    grid_x, grid_y = np.meshgrid(xv, yv)

    nSamples = grid_x.shape[0] * grid_x.shape[1]
    nFeatures = modelDS[0] / shrink * modelDS[1] / shrink * 10
    nChns = sum(chns.info.nChns)

    vFeatures = np.zeros((nSamples, nFeatures), dtype=np.float)
    hs = np.zeros((nSamples, 1), dtype=np.float)

    for idx, (i, j) in enumerate(zip(grid_x.flatten(), grid_y.flatten())):
        src_img = img[j:j+256, i:i+256,:]
        src_ftr = matFtr[:,j/shrink:(j+modelDS[0])/shrink, i / shrink:(i + modelDS[1])/shrink]
        vFeatures[idx] = src_ftr.flatten()
        if not i % 16:
            print i


    h = adaBoostApply(vFeatures, model)

    print "detection end!"

    pos = []
    neg = []
    thr = 10

    for (i, j) in zip(grid_x.flatten().reshape(-1,1)[h > thr], grid_y.flatten().reshape(-1,1)[h > thr]):
        pos.append(img[j:j+256, i:i+256])
    # for (i, j) in zip(grid_x.flatten().reshape(-1,1)[-(h > thr)], grid_y.flatten().reshape(-1,1)[-(h > thr)]):
    #     neg.append(img[j:j+256, i:i+256])

    print len(pos)
    if pos:
        plt.imshow(mosaic(10, pos))
        plt.show()
