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


def get_context(img, pos, sz, window):
    # print img.shape
    # print window.shape
    xs = np.floor(np.arange(pos[0] - sz[0] / 2, pos[0] + sz[0] / 2))
    ys = np.floor(np.arange(pos[1] - sz[1] / 2, pos[1] + sz[1] / 2))
    xs[xs < 1] = 1
    ys[ys < 1] = 1	
    xs[xs > (img.shape[1] - 1)] = img.shape[1] - 1
    ys[ys > (img.shape[0] - 1)] = img.shape[0] - 1
    xs = xs.astype(int)
    ys = ys.astype(int)
    rs = np.dot(ys.reshape(-1, 1), np.ones(len(xs)).reshape(1, -1))
    cs = np.dot(xs.reshape(-1, 1), np.ones(len(ys)).reshape(1, -1)).T
    out = img[rs.astype('int'), cs.astype('int')]

    cv2.namedWindow('Test', 0)
    cv2.imshow('Test', out)

    out = np.float32(out)
    
    out = window * out

    return out


class STC:
    def __init__(self, frame, rect):
        ##initialization
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2 - x1, y2 - y1])
        self.pos = (x1 + x2) // 2, (y1 + y2) // 2
        self.pos_init = (x1 + x2) // 2, (y1 + y2) // 2
        self.target_sz = w, h

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
        self.conff = np.fft.fft2(self.conf)

        # control precision
        self.conff.real = np.float32(self.conff.real)
        self.conff.imag = np.float32(self.conff.imag)
        
        # store pre-computed weight window
        self.ham_win = np.dot(np.hamming(self.sz[1]).reshape(-1, 1),
                              np.hanning(self.sz[0]).reshape(1, -1))
        self.sigma = np.mean(self.target_sz)
        self.window = self.ham_win * np.exp(-0.5 / (self.sigma ** 2) * self.dist)
        self.window = self.window / np.sum(self.window)

        # control precision
        self.window = np.float32(self.window)
        

        self.contextprior = get_context(frame, self.pos, self.sz, self.window)
        self.hscf = self.conff / np.fft.fft2(self.contextprior)
        self.Hstcf = self.hscf

        self.updata_scale()
        self.updata_tracker(frame)

    def updata_scale(self):
        self.sigma = self.sigma * self.scale
        self.window = self.ham_win * np.exp(-0.5 / (self.sigma ** 2) * self.dist)
        self.window = self.window / np.sum(self.window)

    def updata_tracker(self, frame):
        self.contextprior = get_context(frame, self.pos, self.sz, self.window)
        self.confmap = np.fft.ifft2(self.Hstcf * np.fft.fft2(self.contextprior)).real
        pos_now = np.where(self.confmap == self.confmap.max())
        self.pos = self.pos[0] - 0.5 * self.sz[0] + pos_now[1][0] + 1, self.pos[1] - 0.5 * self.sz[1] + pos_now[0][
            0] + 1
        self.cal_psr(pos_now)
        self.contextprior = get_context(frame, self.pos, self.sz, self.window)
        self.hscf = self.conff/np.fft.fft2(self.contextprior) # ??
        self.Hstcf = (1 - self.rho) * self.Hstcf + self.rho * self.hscf

    def draw_state(self, vis): 
        (x, y), (w, h) = self.pos, self.target_sz
        x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.line(vis, (int(x) - 5, int(y)), (int(x) + 5, int(y)), (0, 0, 255), 1)
        cv2.line(vis, (int(x), int(y) - 5), (int(x), int(y) + 5), (0, 0, 255), 1)
        draw_str(vis, (x1, y2 + 16), 'COOR: %.2f %.2f' % self.pos)
        draw_str(vis, (x1, y2 + 30), 'PSR: %.2f' % self.psr)

    def cal_psr(self, pos):
        mval, smean, sstd = self.confmap.max(), self.confmap.mean(), self.confmap.std()
        self.psr = (mval - smean) / (sstd + eps)
        

class App:
    def __init__(self, video_src, paused=False):
        self.cap = video.create_capture(video_src)
        _, self.frame = self.cap.read()
        self.frame = cv2.resize(self.frame, (640, 320))
        # self.video_src = video_src
        # self.frames = os.listdir(video_src)
        # print self.frames
        # self.frame = cv2.imread(os.path.join(video_src, self.frames[0]))
        print 'FrameShape: ', self.frame.shape
        cv2.imshow('Video', self.frame)
        self.rect_sel = RectSelector('Video', self.onrect)
        self.paused = paused
        self.trackers = []

    def onrect(self, rect):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tracker = STC(frame_gray, rect)
        self.trackers.append(tracker)  # multi-target

    def run(self):
        i = 0
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                self.frame = cv2.resize(self.frame, (640, 320))
                # self.frame = cv2.imread(os.path.join(self.video_src, self.frames[i]))
                # i += 1
                if not ret:
                    print 'Get frame fail!'
                    break
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                for tracker in self.trackers:
                    tracker.updata_tracker(frame_gray)

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


if __name__ == '__main__':
    # video_src = r'E:\PROGRAM\APC\sample_test\2'
    video_src = 'H:/2017-03-04-09-15-20/hiv00003.mp4'
    # video_src = '0'
    App(video_src).run()
