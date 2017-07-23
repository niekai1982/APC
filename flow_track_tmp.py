# coding : utf-8
# FileName : STC.py

'''

Fast Tracking via Spatio-Temporal Context Learning

'''

import os
import re
import cv2
import video
import numpy as np
import matplotlib.pyplot as plt
from common import draw_str, RectSelector
import time
# from mpl_toolkits.mplot3d import Axes3D

eps = 1e-20

lk_params = dict( winSize  = (4, 4),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def bb_predict(bb, pt0, pt1, idxF):
    of = pt1 - pt0
    dx = median(of(1,:));
    dy = median(of(2,:));

    d1 = pdist(pt0
    ','
    euclidean
    ');
    d2 = pdist(pt1
    ','
    euclidean
    ');
    s = median(d2. / d1);

    s1 = 0.5 * (s - 1) * bb_width(BB0);
    s2 = 0.5 * (s - 1) * bb_height(BB0);

    BB1 = [BB0(1) - s1;
    BB0(2) - s2;
    BB0(3) + s1;
    BB0(4) + s2] + [dx;
    dy;
    dx;
    dy];
    shift = [s1;
    s2];
    pass


def bb_center(bb):
    bb_x, bb_y = bb[:2]
    bb_w, bb_h = bb[2:]
    bb_center = bb_x + bb_w // 2, bb_y + bb_h // 2
    return bb_center


def bb_points(bb, numM, numN):
    # Generates numMxnumN points on BBox.
    bb_x, bb_y = bb[:2]
    bb_w, bb_h = bb[2:]

    if numM == 1 and numN == 1:
        pt = bb_center(bb)
        return pt

    if numM == 1 and numN > 1:
        c = bb_center(bb)
        stepW = bb_w / (numN - 1)
        pt = np.mgrid[bb_x:(bb_x + bb_w):stepW, c[1]:(c[1] + 1)].astype(np.int)
        return pt

    if numM > 1 and numN == 1:
        c = bb_center(bb)
        stepH = bb_h / (numM - 1)
        pt = np.mgrid[c[1]:(c[1] + 1), bb_y:(bb_y + bb_h):stepH].astype(np.int)
        return pt

    stepW = int(bb_w / (numN - 1))
    stepH = int(bb_h / (numM - 1))
    pt = np.mgrid[bb_x:(bb_x + bb_w):stepW, bb_y:(bb_y + bb_h):stepH].astype(np.int)
    pt = np.dstack((pt[0], pt[1]))
    pt.shape = -1, 1, 2
    return pt


def euclideanDistance(pt0, pt1):
    return np.sqrt(np.square(pt0 - pt1).sum(axis=2))


class OPF:
    def __init__(self, frame, rect):
        ##initialization
        x1, y1, x2, y2 = rect
        w, h = [x2 - x1, y2 - y1]
        self.pos = (x1 + x2) // 2, (y1 + y2) // 2
        self.pos_init = (x1 + x2) // 2, (y1 + y2) // 2
        self.target_sz = w, h
        self.pt0 = bb_points([x1, y1, w, h], numM=50, numN=50)

        # self.updata_tracker(frame)

    def updata_tracker(self, frame_prev, frame):
        p0 = self.pt0
        p0 = p0.astype(np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame, frame_prev, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(frame_prev, frame, p1, None, **lk_params)
        FB = euclideanDistance(p0, p1)
        medFB = np.median(FB)
        idxF = FB >= medFB
        self.pos = bb_predict(self.pt0, p0, p1, idxF)

        # d = abs(p0-p0r).reshape(-1, 2).max(-1)
        # good = d < 1
        # self.pt1 = p1

    def draw_state(self, vis): 
        (x, y), (w, h) = self.pos, self.target_sz
        x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.line(vis, (int(x) - 5, int(y)), (int(x) + 5, int(y)), (0, 0, 255), 1)
        cv2.line(vis, (int(x), int(y) - 5), (int(x), int(y) + 5), (0, 0, 255), 1)
        self.draw_pt(vis)
        draw_str(vis, (x1, y2 + 16), 'COOR: %.2f %.2f' % self.pos)

    def cal_psr(self, pos):
        mval, smean, sstd = self.confmap.max(), self.confmap.mean(), self.confmap.std()
        self.psr = (mval - smean) / (sstd + eps)

    def draw_pt(self, vis):
        for i in range(self.pt0.shape[0]):
            cv2.circle(vis, (self.pt0[i,:,0], self.pt0[i,:,1]), 2, (0, 255, 0), -1)
        for i in range(self.pt1.shape[0]):
            cv2.circle(vis, (self.pt1[i,:,0], self.pt1[i,:,1]), 2, (255, 0, 0), -1)


class App:
    def __init__(self, video_src, paused=False):
        self.cap = video.create_capture(video_src)
        _, self.frame = self.cap.read()
        self.frame = cv2.resize(self.frame, (640, 320))
        # self.video_src = video_src
        # self.frames = os.listdir(video_src)
        # print self.frames
        # self.frame = cv2.imread(os.path.join(video_src, self.frames[0]))
        cv2.imshow('Video', self.frame)
        self.rect_sel = RectSelector('Video', self.onrect)
        self.paused = paused
        self.trackers = []

    def onrect(self, rect):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tracker = OPF(frame_gray, rect)
        self.trackers.append(tracker)  # multi-target

    def run(self):
        i = 0
        # frame_prev = []
        while True:
            if not self.paused:
                self.frame_prev = self.frame
                ret, self.frame = self.cap.read()
                self.frame = cv2.resize(self.frame, (640, 320))
                # self.frame = cv2.imread(os.path.join(self.video_src, self.frames[i]))
                # i += 1
                if not ret:
                    print 'Get frame fail!'
                    break
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                frame_gray_prev = cv2.cvtColor(self.frame_prev, cv2.COLOR_BGR2GRAY)
                for tracker in self.trackers:
                    tracker.updata_tracker(frame_gray, frame_gray_prev)

            vis = self.frame.copy()
            self.rect_sel.draw(vis)

            for tracker in self.trackers:
                tracker.draw_state(vis)

            cv2.imshow('Video', vis)

            ch = cv2.waitKey(1)
            if ch == 27:
                cv2.destroyAllWindows()
                # self.cap.release()
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []
                pass
            i += 1


if __name__ == '__main__':
    # video_src = r'E:\PROGRAM\APC\sample_test\2'
    video_src = 'H:/2017-03-04-09-15-20/hiv00003.mp4'
    # video_src = '0'
    App(video_src).run()
