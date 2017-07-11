import numpy as np
from adaBoostTrain import pModel
from binaryTreeTrain import pData



def forestInds(data, thrs, fids, child):
    # could be optimiz
    N, F = data.shape

    child = child.flatten()
    fids = fids.flatten()
    thrs = thrs.flatten()

    inds = np.zeros((N, 1), dtype=np.int)

    for i in range(N):
        k = 0
        while child[k]:
            if data[i, fids[k]] < thrs[k]:
                k = child[k]
            else:
                k = child[k] + 1
        inds[i] = k
    return inds


def adaBoostApply(data, model):
    nWeak = 256

    N, F = data.shape

    fids = model.fids
    thrs = model.thrs
    child = model.child
    hs = model.hs

    h = np.zeros((N, 1), dtype=np.float)
    for idx_t in range(nWeak):
        inds = forestInds(data, thrs[:,idx_t], fids[:,idx_t], child[:,idx_t])
        h = h + hs[:,idx_t][inds.flatten()].reshape(-1,1)
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

    h0 = adaBoostApply(data.X0, model)
    h1 = adaBoostApply(data.X1, model)

    plt.plot(h0, c='b')
    plt.plot(h1, c='r')
    plt.show()





