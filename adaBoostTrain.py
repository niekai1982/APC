import numpy as np
from binaryTreeTrain import pTree, pData, binaryTreeTrain
from binaryTreeApply import forestInds
import cPickle
import copy
from time import time


class pBoost(object):
    def __init__(self, **data):
        self.__dict__.update(data)


class pModel(object):
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

    msg = 'Training AdaBoost: nWeak=%d nFtrs=%d pos=%d neg=%d'
    print  msg % (nWeak, F, N1, N0)

    H0 = np.zeros((N0, 1), dtype=np.float32)
    H1 = np.zeros((N1, 1), dtype=np.float32)
    losses = np.zeros((1, nWeak), dtype=np.float32)
    errs = losses.copy()
    trees = []

    # main loop
    for i in range(nWeak):
        # train tree and classify each example
        start = time()
        tree, data, err = binaryTreeTrain(data, tree)
        end = time()
        print "binaryTreeTrain per weak spend time = %f" % (end - start)

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

        trees.append(copy.copy(tree))
        errs[0, i] = err
        losses[0, i] = loss

        msg = "i=%4i, alpha=%.3f, err=%.3fm loss=%.2e"
        print msg % (i, alpha, err, loss)

        if loss < 1e-40:
            nWeak = i
            print "stopping early"
            break


    # create output model struct
    k = 0
    for i in range(nWeak):
        k = max(k, trees[i].fids.shape[0])

    model = pModel()

    model.fids = np.zeros((k, nWeak), dtype=np.uint32)
    model.thrs = np.zeros((k, nWeak), dtype=np.float32)
    model.child = np.zeros((k, nWeak), dtype=np.uint32)
    model.hs = np.zeros((k, nWeak), dtype=np.single)
    model.weights = np.zeros((k, nWeak), dtype=np.single)
    model.depth = np.zeros((k, nWeak), dtype=np.uint32)
    model.errs = errs
    model.losses = losses

    for i in range(nWeak):
        T = trees[i]
        k = T.fids.shape[0]
        model.fids[:k,i] = T.fids.T
        model.thrs[:k, i] = T.thrs.T
        model.child[:k, i] = T.child.T
        model.hs[:k, i] = T.hs.T
        model.weights[:k, i] = T.weights.T
        model.depth[:k, i] = T.depth.T
    depth = model.depth.max()
    model.treeDepth = depth * np.all(model.depth[np.logical_not(model.child)] == depth)

    # output info to log
    msg = 'Done training err=%.4f fp=%.4f fn=%.4f (t=%.1fs).'

    return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io as sio

    # load test data
    data_test_X0 = sio.loadmat("C:/Users/nieka/Desktop/test/data_test_X0.mat")
    data_test_X1 = sio.loadmat("C:/Users/nieka/Desktop/test/data_test_X1.mat")
    # data_src = sio.loadmat("C:/Users/nieka/Desktop/test/src_data.mat")

    data = pData()
    data.X0 = data_test_X0['X0'][:, :]
    data.X1 = data_test_X1['X1'][:, :]
    # data.X0 = data_src['data']['X0'][0, 0][:, :]
    # data.X1 = data_src['data']['X1'][0, 0][:, :]
    data.wts0 = np.array([], np.float)
    data.wts1 = np.array([], np.float)
    data.xMin = []
    data.xStep = []
    data.xType = []

    # init boost param
    boost = pBoost()

    boost.nWeak = 128
    boost.discrete = 1
    boost.verbose = 16

    # init tree param
    boost.tree = pTree()
    boost.tree.nBins = 256
    boost.tree.maxDepth = 2
    boost.tree.minWeight = .01
    boost.tree.fracFtrs = 1
    boost.tree.nThreads = 16

    start = time()
    model = adaBoostTrain(data, boost)
    end = time()
    print "adaBoostTrain spend time = %f" % (end - start)

    with open('model.pkl', 'wb') as fp:
        cPickle.dump(model, fp)
        fp.close()
