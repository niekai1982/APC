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
    with open('./data/flow_data/flow_30.pkl', 'rb') as fp:
        flow = cPickle.load(fp)
        fp.close()
    img = cv2.imread('./data/flow_data/img_30.jpg')
    img_prev = cv2.imread('./data/flow_data/previmg_30.jpg')

    plt.imshow(img)
    plt.show()

    # print flow.shape
    #
    # plt.imshow(flow[:,:,1])
    # plt.colorbar()
    # plt.show()
    #
    # num_orients = 2

    # M, O = hof(flow)
    #
    # plt.imshow(M)
    # plt.colorbar()
    # plt.show()
    #
    # h, w = M.shape
    # step = 128
    #
    # y, x = np.mgrid[0:h:step, 0:w:step].reshape(2,-1).astype(int)
    # for (i, j) in zip(y, x):
    #     val = M[i:(i + step), j:(j + step)].sum()
    #     if val > 5 * 128 * 128:
    #         img[i:(i + step), j:(j + step),0] = 255
    #
    # plt.imshow(img)
    # plt.show()
    # H = gradient_Hist(M, O, bin=4, nOrients=num_orients, softBin=0, full=0)

    # h, w = H.shape[:2]
    #
    # H[H < 0.5] = 0
    # print H.shape

    # # plt.subplot(1,2,1)
    # # plt.imshow(M)
    # # plt.subplot(1,2,2)
    # # plt.imshow(O)
    # # plt.show()
    # for i in range(num_orients):
    #     plt.subplot(2,num_orients / 2,i+1)
    #     plt.imshow(H[:,:,i])
    #     plt.colorbar()
    # plt.show()
    #
    # for idx_c in range(H.shape[-1]):
    #     for i in range(H.shape[0]):
    #         for j in range(H.shape[1]):
    #             if H[i, j, idx_c]:
    #                 img[i * 4 * 16:(i+1) * 4 * 16, j * 4 * 16:(j+1) * 4 * 16, 0] = 255
    # plt.imshow(img)
    # plt.show()

    # for i in range(8):
    #     plt.imshow(H[:,:,i])
    #     plt.show()

    # modelDS = [50, 50]
    # step = [8, 8]
    # xv = np.arange(0, w - 1 - modelDS[1], step[1])
    # yv = np.arange(0, h - 1 - modelDS[0], step[0])
    #
    # grid_x, grid_y = np.meshgrid(xv, yv)
    # Samples = np.zeros((grid_x.shape[0], grid_x.shape[1],8), dtype=np.float)

    # for idx_bin in range(8):
    #     for (i, j) in zip(grid_x.flatten(), grid_y.flatten()):
    #         Samples[i, j, idx_bin] = H[i:(i + modelDS[0]), j:(j+modelDS[1]), idx_bin].sum()
    #         Samples[i, j, idx_bin] = 1
