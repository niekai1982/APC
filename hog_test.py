# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:33:22 2016

@author: nieka
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows = img.shape[0]
cols = img.shape[1]

hog = cv2.HOGDescriptor(_winSize=(64,64),_blockSize=(16,16),_blockStride=(8,8),_cellSize=(8,8),_nbins=9)

start = time.time()
hog_feature = hog.compute(gray, winStride=(8,8))
end = time.time()

print (end - start)

start = time.time()

grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

d = np.arctan2(grad_y, grad_x)

end = time.time()
print (end - start)
