from __future__ import print_function

import numpy as np
import cv2
import video
import os
import matplotlib.pyplot as plt
from time import time
import cPickle

if __name__ == '__main__':
    import sys
    print(__doc__)

    file_path = 'C:/Users/nieka/Desktop/test_flow/data'
    os.chdir(file_path)

    img_files = os.listdir('.')

    scale = 2

    prev = cv2.imread(img_files[0])
    prev = cv2.resize(prev, (prev.shape[1] / scale, prev.shape[0] / scale))
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    for i in range(len(img_files)):
        img = cv2.imread(img_files[i])
        img = cv2.resize(img, (img.shape[1] / scale, img.shape[0] / scale))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 3)

        # start = time()
        # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
        # end = time()
        # print(end - start)

        out = np.abs(gray - prevgray)
        out[out > 250] = 0
        out[out < 50] = 0
        # plt.imshow(out)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(out)
        # plt.show()

        # plt.imshow(out)
        # plt.show()

        cv2.imshow('flow', out)
        #
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        # if ch == ord('1'):
        #     show_hsv = not show_hsv
        #     print('HSV flow visualization is', ['off', 'on'][show_hsv])
        # if ch == ord('2'):
        #     show_glitch = not show_glitch
        #     if show_glitch:
        #         cur_glitch = img.copy()
        #     print('glitch is', ['off', 'on'][show_glitch])
        prevgray = gray
    cv2.destroyAllWindows()
