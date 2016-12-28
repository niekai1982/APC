import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from adaBoostTrain import getTrainData

class Tree(object):
    def __init__(self):
        self.nBins = 256
        self.maxDepth = 2
        self.minWeight = 0.01
        self.fracFtrs = 1
        self.nTreads = 16

dfs = Tree()
X0, X1 = getTrainData(nFilePath='data0.pkl', pFilePath='data1.pkl')

N0, F = X0.shape
N1, F1 = X1.shape

xMin = np.zeros((1, F))
xStep = np.ones((1, F))
xType = X0.dtype

wts0 = np.ones((N0, 1)) / N0
wts1 = np.ones((N1, 1)) / N1

w = wts0.sum() + wts1.sum()
if (np.abs(w - 1) > 1e-3):
    wts0 = wts0 / w
    wts1 = wts1 / w
    
xMin = np.min((X0.min(axis=0), X1.min(axis=0)), axis=0) - .01
xMax = np.max((X0.max(axis=0), X1.max(axis=0)), axis=0) + .01
xStep = (xMax - xMin) / (dfs.nBins - 1)


