import numpy as np
import cv2
import video
import os
import matplotlib.pyplot as plt
from time import time
import cPickle
import skimage.io as siio
import copy


lk_params = dict( winSize  = (5, 5),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


if __name__ == '__main__':
    files_path = './data/data'
    files = os.listdir(files_path)
    imgs = []
    for idx, file in enumerate(files):
        file_name = os.path.join(files_path, file)
        imgs.append(siio.imread(file_name))
        if idx > 1:
            break
    # img0, img1, img2 = data
    grays= [cv2.cvtColor(elem, cv2.COLOR_RGB2GRAY) for elem in imgs]

    h, w = grays[0].shape[:2]
    cell_w, cell_h = 128, 128
    cell_step = 8
    grid_step = 16

    # cell init
    # cell_y, cell_x = np.mgrid[0:(h - cell_h):cell_step, 0:(w - cell_w):cell_step].reshape(2, -1).astype(int)
    # cell_coor = np.dstack((cell_y, cell_x))
    cell_x, cell_y = 528, 360
    cell_center = [cell_x + cell_w / 2, cell_y + cell_h / 2]
    cell_start = [cell_x + cell_w / 2, cell_y + cell_h / 2]
    cell_end = [cell_x + cell_w / 2, cell_y + cell_h / 2]

    # p0 init
    p0_y, p0_x = np.mgrid[cell_y:cell_y + cell_h:grid_step, cell_x:cell_x + cell_h:grid_step].reshape(2, -1).astype(np.float32)
    p0 = np.dstack((p0_x, p0_y))
    p0.shape = -1, 1, 2

    p1, st, err = cv2.calcOpticalFlowPyrLK(grays[1], grays[0], p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(grays[0], grays[1], p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1

    p0_v = p0[good]
    p1_v = p1[good]

    c0_y, c0_x = p0_v[:,:,0].mean(), p0_v[:,:,1].mean()
    c1_y, c1_x = p1_v[:,:,0].mean(), p1_v[:,:,1].mean()

    cell_y, cell_x = c1_y - cell_h / 2, c1_x - cell_w / 2

    p0_y, p0_x = np.mgrid[cell_y:cell_y + cell_h:grid_step, cell_x:cell_x + cell_h:grid_step].reshape(2, -1).astype(np.float32)
    p0 = np.dstack((p0_x, p0_y))
    p0.shape = -1, 1, 2

    p1, st, err = cv2.calcOpticalFlowPyrLK(grays[2], grays[1], p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(grays[1], grays[2], p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1

    p0_v = p0[good]
    p1_v = p1[good]

    # c0_y, c0_x = p0_v[:,:,0].mean(), p0_v[:,:,1].mean()
    c2_y, c2_x = p1_v[:,:,0].mean(), p1_v[:,:,1].mean()



    plt.imshow(imgs[1])

    plt.scatter(c0_y, c0_x, c='r', marker='+')
    plt.scatter(c1_y, c1_x, c='b', marker='+')
    plt.scatter(c2_y, c2_x, c='g', marker='+')
    # plt.scatter(p0_v[:,:,0], p0_v[:,:,1], c='r', marker='+')
    # plt.scatter(p1_v[:,:,0], p1_v[:,:,1], c='b', marker='+')
    plt.show()
    # plt.scatter(cell_coor[:,:,0], cell_coor[:,:,1], marker='+', c='r')
    # plt.scatter(cell_start[:,:,0], cell_start[:,:,1], marker='+', c='b')
    # plt.show()

