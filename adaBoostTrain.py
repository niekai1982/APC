import numpy as np
from binaryTreeTrain import pTree, pData, binaryTreeTrain
from binaryTreeApply import forestInds


class pBoost(object):
    def __init__(self, **data):
        self.__dict__.update(data)


def adaBoostTrain(data, boost):
    X0 = data.X0
    X1 = data.X1

    N0, F = X0.shape
    N1, F1 = X1.shape
    assert F == F1

    nWeak = boost.nWeak
    discrete = boost.discrete
    verbose = boost.verbose
    tree = boost.tree

    print 'Training AdaBoost: nWeak=%d nFtrs=%d pos=%d neg=%d' % (nWeak, F, N1, N0)
    H0 = np.zeros((N0, 1), dtype=np.float32)
    H1 = np.zeros((N1, 1), dtype=np.float32)
    losses = np.zeros((1, nWeak), dtype=np.float32)

    # main loop
    for i in range(2):
        # train tree and classify each example
        tree, data, err = binaryTreeTrain(data, tree)
        if discrete:
            tree.hs = (tree.hs > 0) * 2 - 1
        h0 = forestInds(X0, tree)
        h1 = forestInds(X1, tree)

        # compute alpha and incorporate directly into tree model
        alpha = 1
        if discrete:
            alpha = max(-1, min(5, .5 * np.log((1 - err) / err)))
        if verbose and alpha <= 0:
            nWeak = i - 1
            print 'stopping early'
            break
        tree.hs = tree.hs * alpha

        # update comulative soures H and weights
        H0 = H0 + h0 * alpha
        data.wts0 = np.exp(H0) / N0 / 2
        H1 = H1 + h1 * alpha
        data.wts1 = np.exp(-H1) / N1 / 2
        loss = data.wts0.sum() + data.wts1.sum()

    return loss


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io as sio

    # load test data
    data_src = sio.loadmat("C:/Users/nieka/Desktop/test/src_data.mat")

    data = pData()
    data.X0 = data_src['data']['X0'][0, 0][:, :]
    data.X1 = data_src['data']['X1'][0, 0][:, :]
    data.wts0 = np.array([], np.float)
    data.wts1 = np.array([], np.float)
    data.xMin = []
    data.xStep = []
    data.xType = []

    # init boost param
    boost = pBoost()

    boost.nWeak = 1
    boost.discrete = 1
    boost.verbose = 16

    # init tree param
    boost.tree = pTree()
    boost.tree.nBins = 256
    boost.tree.maxDepth = 2
    boost.tree.minWeight = .01
    boost.tree.fracFtrs = 1
    boost.tree.nThreads = 16

    h0, h1 = adaBoostTrain(data, boost)
