from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.datasets import make_gaussian_quantiles

"""
INPUTS
  data       - data for training tree
   .X0         - [N0xF] negative feature vectors
   .X1         - [N1xF] positive feature vectors
   .wts0       - [N0x1] negative weights
   .wts1       - [N1x1] positive weights
   .xMin       - [1xF] optional vals defining feature quantization
   .xStep      - [1xF] optional vals defining feature quantization
   .xType      - [] optional original data type for features
  pTree      - additional params (struct or name/value pairs)
   .nBins      - [256] maximum number of quanizaton bins (<=256)
   .maxDepth   - [1] maximum depth of tree
   .minWeight  - [.01] minimum sample weigth to allow split
   .fracFtrs   - [1] fraction of features to sample for each node split
   .nThreads   - [16] max number of computational threads to use

 OUTPUTS
  tree       - learned decision tree model struct w the following fields
   .fids       - [Kx1] feature ids for each node
   .thrs       - [Kx1] threshold corresponding to each fid
   .child      - [Kx1] index of child for each node (1-indexed)
   .hs         - [Kx1] log ratio (.5*log(p/(1-p)) at each node
   .weights    - [Kx1] total sample weight at each node
   .depth      - [Kx1] depth of each node
  data       - data used for training tree (quantized version of input)
  err        - decision tree training error
"""

esp = 1e-6


class pTree(object):
    def __init__(self, **data):
        self.__dict__.update(data)


class pData(object):
    def __init__(self, **data):
        self.__dict__.update(data)


def binaryTreeTrain(data, Tree):
    """

    :type pTree: pTree
    :type data: pData
    """
    # Intial Data Struct
    X0 = data.X0
    X1 = data.X1
    wts0 = data.wts0
    wts1 = data.wts1
    xMin = data.xMin
    xStep = data.xStep
    xType = data.xType

    # Initial Tree Struct
    nBins = Tree.nBins
    maxDepth = Tree.maxDepth
    minWeight = Tree.minWeight
    fracFtrs = Tree.fracFtrs
    nThreads = Tree.nThreads

    # get data and normalize weights
    N0, F = X0.shape
    N1, F1 = X1.shape
    assert F == F1

    if not xType:
        xMin = np.zeros((1, F))
        xStep = np.ones((1, F))
        xType = np.double

    assert wts0.dtype == np.float
    assert wts0.dtype == np.float

    if not wts0:
        wts0 = np.ones((N0, 1)) / N0
        wts1 = np.ones((N1, 1)) / N1

    w = wts0.sum() + wts1.sum()

    if (abs(w - 1) > 1e-3):
        wts0 = wts0 / w
        wts1 = wts1 / w

    # quantize data to be between [0, nBins-1] if not already quantized
    if X0.dtype != np.uint8 or X0.dtype != np.uint8:
        xMin = np.minimum(X0.min(0), X1.min(0)) - .01
        xMax = np.maximum(X0.max(0), X1.max(0)) + .01
        xStep = (xMax - xMin) / (nBins - 1)
        X0 = np.uint8(np.round((X0 - xMin) / xStep))
        X1 = np.uint8(np.round((X1 - xMin) / xStep))

    # train decision tree classifier
    K = 2 * (N0 + N1)

    errs = np.zeros((K, 1), xType)
    thrs = np.zeros((K, 1), xType)

    hs = np.zeros((K, 1), dtype=np.single)
    weights = hs.copy()

    fids = np.zeros((K, 1), dtype=np.uint32)
    child = fids.copy()
    depth = fids.copy()

    wtsAll0 = [None] * K
    wtsAll0[0] = wts0
    wtsAll1 = [None] * K
    wtsAll1[0] = wts1

    k = 0
    K = 1
    while k < K:

        # get node weights and prior
        wts0 = wtsAll0[k]
        wtsAll0[k] = None
        w0 = wts0.sum()
        wts1 = wtsAll1[k]
        wtsAll1[k] = None
        w1 = wts1.sum()

        w = w0 + w1
        prior = w1 / w

        weights[k] = w
        errs[k] = min(prior, 1 - prior)
        hs[k] = max(-4, min(4, .5 * np.log((prior + esp) / (1 - prior + esp))))

        if prior < 1e-3 or prior > 1 - 1e-3 or depth[k] >= maxDepth or w < minWeight:
            k = k + 1
            continue

        fidsSt = range(F)

        if fracFtrs < 1:
            fidsSt = np.random.permutation(F)[:int(np.floor(F * fracFtrs))]

        errsSt, thrsSt = binaryTreeTrain1(X0, X1, np.single(wts0 / w), np.single(wts1 / w),
                                          nBins, prior, fidsSt, nThreads)
        # fid = np.argsort(errsSt, axis=0)[0]
        fid = np.where(errsSt == errsSt.min())[0][0]
        thr = np.single(thrsSt[fid] + .5)
        # err: converting an array with ndim > 0 to an index will result in an error in the futurer
        fid = fidsSt[fid]

        left0 = X0[:, fid] < thr
        left1 = X1[:, fid] < thr

        if (np.any(left0) or np.any(left1)) and (np.any(~left0) or np.any(~left1)):
            thr = xMin[fid] + xStep[fid] * thr
            child[k] = K
            fids[k] = fid
            thrs[k] = thr
            wtsAll0[K] = wts0 * left0.reshape(-1, 1)
            wtsAll0[K + 1] = wts0 * ~left0.reshape(-1, 1)
            wtsAll1[K] = wts1 * left1.reshape(-1, 1)
            wtsAll1[K + 1] = wts1 * ~left1.reshape(-1, 1)
            depth[K:K + 2] = depth[k] + 1
            K = K + 2
        k = k + 1
    K = K - 1
    Tree.fids = fids[:K + 1]
    Tree.thrs = thrs[:K + 1]
    Tree.child = child[:K + 1]
    Tree.hs = hs[:K + 1]
    Tree.weights = weights[:K + 1]
    Tree.depth = depth[:K + 1]
    err = errs[:K + 1] * Tree.weights * (Tree.child == 0)
    return Tree, data, err.sum()


def binaryTreeTrain1(X0, X1, wts0, wts1, nBins, prior, fidsSt, nThreads):
    N0, F = X0.shape
    N1, F1 = X1.shape
    assert F == F1

    errs = np.empty((F, 1), dtype=np.float)
    thrs = np.empty((F, 1), dtype=np.uint8)

    for f in fidsSt:
        cdf0 = np.zeros((nBins, 1), dtype=np.float)
        cdf1 = np.zeros((nBins, 1), dtype=np.float)
        thr = 0
        if prior < .5:
            e0 = prior
            e1 = 1 - prior
        else:
            e0 = 1 - prior
            e1 = prior
        for i in range(N0):
            cdf0[X0[i, f]] += wts0[i]
        for i in range(N1):
            cdf1[X1[i, f]] += wts1[i]
        for i in range(1, nBins):
            cdf0[i] += cdf0[i - 1]
            cdf1[i] += cdf1[i - 1]
        for i in range(nBins):
            e = prior - cdf1[i] + cdf0[i]

            # err find?
            if (e0 - e) > esp:
                e0 = e
                e1 = 1 - e
                thr = i
            elif (e - e1) > esp:
                e0 = 1 - e
                e1 = e
                thr = i

        errs[f] = e0
        thrs[f] = thr
    return errs, thrs






if __name__ == '__main__':
    from binaryTreeApply import forestInds
    import pickle
    print 'start ----->'

    # data_src, label = make_blobs(n_samples=1000, n_features=2, centers=2)
    # X1, y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2, n_classes=2, random_state=1)
    # X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)
    # X = np.concatenate((X1, X2))
    # y = np.concatenate((y1, -y2 + 1))
    #
    # plt.scatter(X[:, 0], X[:, 1], c=y, marker='+')
    # plt.show()
    #
    data_src = sio.loadmat("C:/Users/nieka/Desktop/test/src_data.mat")
    data = pData()
    data.X0 = data_src['data']['X0'][0, 0][:, :]
    data.X1 = data_src['data']['X1'][0, 0][:, :]

    print data.X0.shape
    print data.X1.shape

    data.wts0 = np.array([], np.float)
    data.wts1 = np.array([], np.float)
    data.xMin = []
    data.xStep = []
    data.xType = []

    tree = pTree()
    tree.nBins = 256
    tree.maxDepth = 2
    tree.minWeight = .01
    tree.fracFtrs = 1
    tree.nThreads = 16

    # train binaryTree
    print "======= train start ======="
    tree, data, err = binaryTreeTrain(data, tree)
    out_put = open('binary_tree_model.pkl','wb')
    pickle.dump(tree, out_put)
    out_put.close()
    print "======= train end ======="

    # print "======= model test ======="
    # # x = data.X0[2,:]
    # # x.shape = 1, -1
    # inds = forestInds(data.X0, tree)
    # print inds

    print "end<--------"
