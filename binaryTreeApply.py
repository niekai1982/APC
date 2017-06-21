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
    fids = tree.fids.flatten()
    thrs = tree.thrs.flatten()

    inds = np.zeros((N, 1), dtype=np.int)

    for i in range(N):
        k = 0
        print "idx now is : %d" % i
        while child[k]:
            if data[i, fids[k]] < thrs[k]:
                k = child[k] - 1
            else:
                k = child[k]
        inds[i] = k + 1
    return inds.flatten()


if __name__ == '__main__':
    print 'test'
    print 'test'
    pass
