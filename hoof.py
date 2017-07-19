import numpy as np
import cv2
import video
import os
import matplotlib.pyplot as plt
from time import time
import cPickle
from getFeature_test import gradient_Hist
from getFeature_test import gradient_Mag
# import numpy.core.multiarray


def hof(flow):
    Gx, Gy = flow[:,:,0], flow[:,:,1]
    M, O = cv2.cartToPolar(Gx, Gy, angleInDegrees=False)
    O[O > np.pi] -= np.pi
    return M, O


if __name__ == '__main__':
    with open('./data/flow_data/flow_301.pkl', 'rb') as fp:
        flow = cPickle.load(fp)
        fp.close()

    num_orients = 6

    M, O = hof(flow)
    H = gradient_Hist(M, O, bin=4, nOrients=num_orients, softBin=0, full=0)

    h, w = H.shape[:2]

    # # plt.subplot(1,2,1)
    # # plt.imshow(M)
    # # plt.subplot(1,2,2)
    # # plt.imshow(O)
    # # plt.show()
    for i in range(num_orients):
        plt.subplot(2,num_orients / 2,i+1)
        plt.imshow(H[:,:,i])
        plt.colorbar()
    plt.show()

    # for i in range(8):
    #     plt.imshow(H[:,:,i])
    #     plt.show()

    modelDS = [50, 50]
    step = [8, 8]
    xv = np.arange(0, w - 1 - modelDS[1], step[1])
    yv = np.arange(0, h - 1 - modelDS[0], step[0])

    grid_x, grid_y = np.meshgrid(xv, yv)
    Samples = np.zeros((grid_x.shape[0], grid_x.shape[1],8), dtype=np.float)

    for idx_bin in range(8):
        for (i, j) in zip(grid_x.flatten(), grid_y.flatten()):
            Samples[i, j, idx_bin] = H[i:(i + modelDS[0]), j:(j+modelDS[1]), idx_bin].sum()
            Samples[i, j, idx_bin] = 1
