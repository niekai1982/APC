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
from mpl_toolkits.mplot3d import Axes3D
eps = 1e-5

def get_context(img, pos, sz, window):
    # print img.shape
    # print window.shape
    xs = np.floor(np.arange(pos[0] - sz[0] / 2, pos[0] + sz[0] / 2))
    ys = np.floor(np.arange(pos[1] - sz[1] / 2, pos[1] + sz[1] / 2))
    print len(xs), len(ys)
    xs[xs < 1] = 1
    ys[ys < 1] = 1
    xs[xs > (img.shape[1] - 1)] = img.shape[1] - 1
    ys[ys > (img.shape[0] - 1)] = img.shape[0] - 1
    xs = xs.astype(int)
    ys = ys.astype(int)
    rs = np.dot(ys.reshape(-1, 1), np.ones(len(xs)).reshape(1, -1))
    cs = np.dot(xs.reshape(-1, 1), np.ones(len(ys)).reshape(1, -1)).T
    # x, y = np.mgrid[xs.min():xs.max(), ys.min():ys.max()]  # ?
    # print x.shape
    # print x, y
    out = img[rs.astype('int'), cs.astype('int')]
    # out = out.T
    cv2.namedWindow('Test', 0)
    cv2.imshow('Test', out)
    # print out
    out = out.astype('double') # np.double(out)
    out = out - np.mean(out)
    out = window * out
    #plt.imshow(out)
    # plt.hold()
    # plt.show()
    # print out
    return out


class STC:
    def __init__(self, frame, rect):
        # initialization
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2 - x1, y2 - y1])
        # x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
        self.pos = (x1 + x2) // 2, (y1 + y2) // 2
        self.target_sz = w, h # w : col; h : row
        print 'POS: ', self.pos
        print 'TargetSize:', self.target_sz

        # parameters according to the paper.
        self.padding = 1  # extra area surrounding the target
        self.rho = 0.075  # the learning parameter \rho
        self.sz = np.floor(np.array(self.target_sz) * (1 + self.padding))  # size of context region

        # parameters of scale updata
        self.scale = 1
        self.lamb = 0.25
        self.num = 5

        # pre-computed confidence map
        self.alapha = 2.25  # parmeter \alpha in Eq.(6)
        self.rs, self.cs = np.mgrid[(1 - self.sz[1] / 2):(1 + self.sz[1] - self.sz[1] / 2),
                                    (1 - self.sz[0] / 2):(1 + self.sz[0] - self.sz[0] / 2)]
        self.dist = self.rs ** 2 + self.cs ** 2
        self.conf = np.exp(-0.5 / self.alapha * np.sqrt(self.dist))
        self.conf = self.conf / np.sum(self.conf)

        self.conff = cv2.dft(self.conf, flags=cv2.DFT_COMPLEX_OUTPUT)

        # store pre-computed weight window
        self.ham_win = np.dot(np.hamming(self.sz[1]).reshape(-1, 1),
                              np.hanning(self.sz[0]).reshape(1, -1))
        self.sigma = np.mean(self.target_sz)
        self.window = self.ham_win * np.exp(-0.5 / (self.sigma ** 2) * self.dist)
        self.window = self.window / np.sum(self.window)


        self.contextprior = get_context(frame, self.pos, self.sz, self.window)
        self.hscf = self.conff / (cv2.dft(self.contextprior,
                                         flags=cv2.DFT_COMPLEX_OUTPUT) + eps)  #??
        self.Hstcf = self.hscf

        self.updata_scale()
        self.updata_tracker(frame)

    def updata_scale(self):
        self.sigma = self.sigma * self.scale
        self.window = self.ham_win * np.exp(-0.5 / (self.sigma ** 2) * self.dist)
        self.window = self.window / np.sum(self.window)

    def updata_tracker(self, frame):
        print 'POS_INIT: x = %f, y = %f' % self.pos
        self.contextprior = get_context(frame, self.pos, self.sz, self.window)
        self.confmap = cv2.idft(self.Hstcf * cv2.dft(self.contextprior, flags=cv2.DFT_COMPLEX_OUTPUT),
                                flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        pos_now = np.where(self.confmap == self.confmap.max())
        self.pos = self.pos[0] - 0.5 * self.sz[0] + pos_now[1][0], self.pos[1] - 0.5 * self.sz[1] + pos_now[0][0]
        cv2.imshow('map', self.confmap)

        print 'POS_TRACKER: x = %f, y = %f' % self.pos


class App:
    def __init__(self, video_src, paused=False):
        self.cap = video.create_capture(video_src)
        _, self.frame = self.cap.read()
        print 'FrameShape: ', self.frame.shape
        cv2.imshow('Video', self.frame)
        self.rect_sel = RectSelector('Video', self.onrect)
        self.paused = paused
        self.trackers = []

    def onrect(self, rect):
        print 'RectSize: ', rect
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tracker = STC(frame_gray, rect)
        self.trackers.append(tracker)  # multi-target

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    print 'Get frame fail!'
                    break
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                for tracker in self.trackers:
                    # print tracker
                    tracker.updata_tracker(frame_gray)

            vis = self.frame.copy()
            self.rect_sel.draw(vis)
            cv2.imshow('Video', vis)

            ch = cv2.waitKey(1)
            if ch == 27:
                cv2.destroyAllWindows()
                self.cap.release()
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                pass


if __name__ == '__main__':
    video_src = 0
    App(video_src).run()
