import numpy as np
from adaBoostTrain import pModel
from binaryTreeTrain import pData


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

    data = pData()
    model = pModel()

    data_src = sio.loadmat("C:/Users/nieka/Desktop/test/src_data.mat")
    data.X0 = data_src['data']['X0'][0, 0][:, :]
    data.X1 = data_src['data']['X1'][0, 0][:, :]

    with open('model.pkl', 'rb') as fp:
        model = cPickle.load(fp)
        fp.close()

    h0 = []
    h1 = []
    for elem in data.X0:
        h0.append(acfDetect(elem, model))
    for elem in data.X1:
        h1.append(acfDetect(elem, model))

    plt.plot(h0, c='r')
    plt.plot(h1, c='b')
    plt.show()

