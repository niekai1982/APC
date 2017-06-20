import numpy as np
from binaryTreeTrain import pTree


def binaryTreeApply(X, tree, maxDepth, minWeight, nThreads):
    if not maxDepth:
        maxDepth = 0
    if not minWeight:
        minWeight = 0
    if not nThreads:
        nThreads = 16
    if maxDepth > 0
        tree.child[tree.depth >= maxDepth] = 0
    if minWeight > 0:
        tree.child[tree.weights <= minWeight] = 0
    pass


def forestInds(data, tree):
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
