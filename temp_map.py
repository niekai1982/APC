import skimage.io as siio
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from common import mosaic
import cv2

def nmsMax(bbs, overlap, greedy, ovrDnm):
# # for each i suppress all j st j > i and area-overlap > overlap

    ord = np.argsort(bbs[:, 4])[::-1]
    # [~, ord] = sort(bbs(:, 5), 'descend');
    bbs = bbs[ord]
    n = bbs.shape[0]
    kp = np.ones(n, dtype=np.bool)
    ars = bbs[:, 2] * bbs[:, 3]
    xs = bbs[:, 0]
    xe = bbs[:, 0] + bbs[:, 2]
    ys = bbs[:, 1]
    ye = bbs[:, 1] + bbs[:, 3]
    for i in range(n):
        if greedy and not kp[i]:
            continue
        for j in range(i+1, n):
            if not kp[j]: continue
            iw = min(xe[i], xe[j]) - max(xs[i], xs[j])
            if iw <= 0: continue
            ih = min(ye[i], ye[j]) - max(ys[i], ys[j])
            if ih <= 0: continue
            o = iw * ih
            if ovrDnm:
                u =ars[i] +ars[j] - o
            else:
                u = min(ars[i],ars[j])
            o = o * 1. / u
            if (o > overlap):
                kp[j] = False
    out = bbs[kp]
    return out


def track():
    pass


if __name__ == '__main__':
    data_path = 'D:/TEST_DATA/test_flow/data'
    map_path = 'D:/TEST_DATA/test_flow/map'

    data_set = os.listdir(data_path)
    map_set = os.listdir(map_path)

    data = []
    map = []

    num2test = 10

    for i in range(500):
        data_name = os.path.join(data_path, data_set[i])
        data.append(siio.imread(data_name))

    for i in range(500):
        map_name = os.path.join(map_path, map_set[i])
        map.append(sio.loadmat(map_name)['bbs'])

    h, w = data[0].shape[:2]
    for i in range(500):
        vis = np.zeros((h, w), dtype=np.float)

        for j in range(map[i].shape[0]):
            rect_x = map[i][j,0]
            rect_y = map[i][j,1]
            rect_w = map[i][j,2]
            rect_h = map[i][j,3]
            rect_score = map[i][j,4]
            rect_centX = int(rect_x + rect_w / 2)
            rect_centY = int(rect_y + rect_h / 2)
            data[i][rect_centY - 1: rect_centY + 2, rect_centX - 1 : rect_centX + 2,0] = rect_score * 2
            data[i][rect_centY - 1: rect_centY + 2, rect_centX - 1 : rect_centX + 2,1:] = 0

        bbs = nmsMax(map[i], overlap=0.55, greedy=1, ovrDnm=0)
        for k in range(bbs.shape[0]):
            rect_x = bbs[k,0]
            rect_y = bbs[k,1]
            rect_w = bbs[k,2]
            rect_h = bbs[k,3]
            rect_score = bbs[k,4]
            rect_centX = int(rect_x + rect_w / 2)
            rect_centY = int(rect_y + rect_h / 2)
            data[i][rect_centY - 4 : rect_centY + 4, rect_centX,:2] = 255
            data[i][rect_centY, rect_centX - 4 : rect_centX + 4,:2] = 255
            # data[i][rect_centY - 8: rect_centY + 8, rect_centX - 8 : rect_centX + 8,:2] = 0
        cv2.imshow('vis', data[i][:,:,::-1])
        ch = 0xFF & cv2.waitKey(100)
        if ch == 27:
            break

        # plt.subplot(1,2,1)
        # plt.imshow(data[i])
        # plt.subplot(1,2,2)
        # plt.imshow(mosaic(2, [vis, data[i][:,:,0]]))
        # plt.imshow(vis)
        # plt.hold()
        # plt.pause(0.0001)


