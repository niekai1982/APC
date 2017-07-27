import skimage.io as siio
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from common import mosaic
import cv2
import copy
import cython

# def test():
#     cdef int i = 1
#     return i

color_set = [(255,0,0), (255,0,255), (255,255,0),(0,0,255),(0,255,0),(0,255,255),(255,255,255),(0,0,0)]
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
    out = []
    for i in range(n):
        if greedy and not kp[i]:
            continue
        tmp_set = bbs[i].copy()
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
                tmp_set = np.vstack((tmp_set, bbs[j]))
        tmp_set.shape = -1, 5
        x = tmp_set[:, 0].mean()
        y = tmp_set[:, 1].mean()
        w = tmp_set[:, 2].mean()
        h = tmp_set[:, 3].mean()
        score = tmp_set[:,4].max()
        out.append(np.array([x + w / 3, y + h / 3, w / 3, h / 3, score]))
    # out = bbs[kp]
    return out


def calcOverlap(rect1, rect2, method='max'):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    ars1 = w1 * h1
    ars2 = w2 * h2
    xs1, xs2 = x1, x2
    xe1, xe2 = x1 + w1, x2 + w2
    ys1, ys2 = y1, y2
    ye1, ye2 = y1 + h1, y2 + h2
    iw = min(xe1, xe2) - max(xs1, xs2)
    if iw <= 0: return 0
    ih = min(ye1, ye2) - max(ys2, ys2)
    if ih <= 0: return 0
    o = iw * ih
    u = min(ars1,ars2)
    o = o * 1. / u
    return o


def track_update(tracker_set, Rect_n, thr, f_idx):
    # tracker_set = copy.copy(trackers)
    kt = np.ones(len(Rect_n), dtype=np.bool)
    if not tracker_set:
        for rect in Rect_n:
            rect = np.append(rect, f_idx)
            rect.shape = 1, 6
            tracker_set.append(rect)
        return tracker_set
    for i_t, tracker in enumerate(tracker_set):
        for i_r, rect in enumerate(Rect_n):
            if not kt[i_r]:continue
            overlap = calcOverlap(tracker[-1][:-2], rect[:-1])
            if overlap > thr and f_idx - tracker[-1][-1] < 4:
                rect = np.append(rect, f_idx)
                rect.shape = 1, 6
                tracker_set[i_t] = np.vstack((tracker, rect))
                kt[i_r] = False
    idx = list(np.where(kt)[0])
    for i in idx:
        tracker_set.append(np.append(Rect_n[i], f_idx).reshape(1, 6))
    out = []
    for i_t, tracker in enumerate(tracker_set):
        t_s = tracker[0,5]
        t_d = tracker[-1,5]
        if f_idx - t_d > 2:
            continue
        out.append(tracker)
    return out

def get_tr(tracker):
    tr = []
    for i in range(tracker.shape[0]):
        x, y, w, h = tracker[i, :4]
        tr.append((x + w / 2, y + h / 2))
    return tr

def draw_probMap(vis, map):
    data = vis.copy()
    for j in range(map.shape[0]):
        rect_x = map[j,0]
        rect_y = map[j,1]
        rect_w = map[j,2]
        rect_h = map[j,3]
        rect_score = map[j,4]
        rect_centX = int(rect_x + rect_w / 2)
        rect_centY = int(rect_y + rect_h / 2)
        data[rect_centY - 1: rect_centY + 2, rect_centX - 1 : rect_centX + 2,0] = rect_score * 2
        data[rect_centY - 1: rect_centY + 2, rect_centX - 1 : rect_centX + 2,1:] = 0
    return data


if __name__ == '__main__':
    data_path = 'D:/TEST_DATA/test_flow/data'
    map_path = 'D:/TEST_DATA/test_flow/map'

    data_set = os.listdir(data_path)
    map_set = os.listdir(map_path)

    data = []
    map = []

    num2test = 10


    tracker = []
    paused = False
    idx_frame = 0
    cv2.namedWindow('vis', 0)
    while idx_frame < 500:
        if not paused:
            data_name = os.path.join(data_path, data_set[idx_frame])
            map_name = os.path.join(map_path, map_set[idx_frame])

            data = siio.imread(data_name)
            map = sio.loadmat(map_name)['bbs']

            h, w = data.shape[:2]

            # vis = np.zeros((h, w), dtype=np.float)
            vis = draw_probMap(data, map)

            head_rect = nmsMax(map, overlap=0.5, greedy=1, ovrDnm=0)
            tracker = track_update(tracker, head_rect, thr=0.4, f_idx=idx_frame)

            # cv2.polylines(vis, [np.int32(get_tr(tr)) for tr in tracker], False, (255,255,255), thickness=2)
            for i, tr in enumerate(tracker):
                if tr[:, -2].max() > 80 and tr.shape[0] > 10:
                    cv2.polylines(vis, [np.int32(get_tr(tr))], False, color_set[i % 8], thickness=2)
            idx_frame += 1

        cv2.imshow('vis', vis[:,:,::-1])
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        if ch == ord(' '):
            paused = not paused

        # plt.subplot(1,2,1)
        # plt.imshow(data[i])
        # plt.subplot(1,2,2)
        # plt.imshow(mosaic(2, [vis, data[i][:,:,0]]))
        # plt.imshow(vis)
        # plt.hold()
        # plt.pause(0.0001)

            # for i in range(500):
            #     data_name = os.path.join(data_path, data_set[i])
            #     # data.append(siio.imread(data_name))
            #
            # for i in range(500):
            #     map_name = os.path.join(map_path, map_set[i])
            #     map.append(sio.loadmat(map_name)['bbs'])


            # cv2.rectangle()

            # for bbs in out:
            #     rect_x = bbs[0]
            #     rect_y = bbs[1]
            #     rect_w = bbs[2]
            #     rect_h = bbs[3]
            #     rect_score = bbs[4]
            #     rect_centX = int(rect_x + rect_w / 2)
            #     rect_centY = int(rect_y + rect_h / 2)
            #     data[i][rect_centY - 4 : rect_centY + 4, rect_centX,:2] = 255
            #     data[i][rect_centY, rect_centX - 4 : rect_centX + 4,:2] = 255
            #         # data[i][rect_centY - 8: rect_centY + 8, rect_centX - 8 : rect_centX + 8,:2] = 0
