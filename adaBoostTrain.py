import numpy as np
import pickle
from binaryTreeTrain import pTree, pData, binaryTreeTrain


class Tree(object):
    def __init__(self):
        self.nBins = 256
        self.maxDepth = 1
        self.minWeight = 0.01
        self.fracFtrs = 1
        self.nThreads = 16

class Adaboost():
    def __init__(self):
        self.pTree = Tree()
        self.nWeak = 128
        self.discrete = 1
        self.verbose = 1

def getTrainData(pFilePath, nFilePath):
    p_dataFile = open(pFilePath, 'rb')
    n_dataFile = open(nFilePath, 'rb')
    X0 = pickle.load(n_dataFile)
    X1 = pickle.load(p_dataFile)
    X0.shape = X0.shape[0], X0.shape[1]
    X1.shape = X1.shape[0], X1.shape[1]
    p_dataFile.close()
    n_dataFile.close()
    return X0, X1

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dfs = Adaboost()
    X0, X1 = getTrainData(pFilePath='data1.pkl', nFilePath='data0.pkl')
    data = {}
    data['X0'] = X0
    data['X1'] = X1
    N0, F = X0.shape
    N1, F1 = X1.shape
    msg = 'Training AdaBoost: nWeak=%d nFtrs=%d pos=%d neg=%d'
    if dfs.verbose:
        print msg % (dfs.nWeak, F, N1, N0)
    H0 = np.zeros((N0, 1))
    H1 = np.zeros((N1, 1))
    losses = np.zeros((1, dfs.nWeak))
    for i in range(dfs.nWeak):
        pass

