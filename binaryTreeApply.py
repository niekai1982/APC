import numpy as np
from binaryTreeTrain import pTree, pData


def binaryTreeApply(X, tree, maxDepth, minWeight, nThreads):
    if not maxDepth:
        maxDepth = 0
    if not minWeight:
        minWeight = 0
    if not nThreads:
        nThreads = 16
    if maxDepth > 0:
        tree.child[tree.depth >= maxDepth] = 0
    if minWeight > 0:
        tree.child[tree.weights <= minWeight] = 0
    hs = tree.hs[forestInds(X, tree)]
    return hs


def forestInds(data, tree):
    # could be optimiz
    N, F = data.shape

    child = tree.child.flatten()
    fids = tree.fids.flatten() - 1
    thrs = tree.thrs.flatten()

    inds = np.zeros((N, 1), dtype=np.int)

    for i in range(N):
        k = 0
        while child[k]:
            if data[i, fids[k]] < thrs[k]:
                k = child[k]
            else:
                k = child[k] + 1
        inds[i] = k + 1
    return inds.flatten()


if __name__ == '__main__':
    import pickle
    import scipy.io as sio

    # load binaryTreeModel
    pkl_file = open('binary_tree_model.pkl', 'rb')
    tree = pickle.load(pkl_file)
    pkl_file.close()

    # load test data
    data_src = sio.loadmat("C:/Users/nieka/Desktop/test/src_data.mat")
    data = pData()
    data.X0 = data_src['data']['X0'][0, 0][:, :]
    data.X1 = data_src['data']['X1'][0, 0][:, :]

    print data.X0.shape
    print data.X1.shape

    out_test = forestInds(data.X0, tree)
    print out_test
