import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

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





