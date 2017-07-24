#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
# from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock
import os
from time import time
import matplotlib.pyplot as plt

lk_params = dict( winSize  = (10, 10),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.2,
                       minDistance = 3,
                       blockSize = 10 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 1
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0

    def run(self):
        while True:
            scale = 2
            ret, frame = self.cam.read()
            frame = cv2.resize(frame, (frame.shape[1] / scale, frame.shape[0] / scale))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                d_t = abs(p0-p1).reshape(-1, 2).max(-1)
                good = d < 1
                dist = d_t < 1
                new_tracks = []
                for tr, (x, y), good_flag, dist_flag in zip(self.tracks, p1.reshape(-1, 2), good, dist):
                    if not good_flag or dist_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                start = time()
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                end = time()
                print end - start
                # step = 16
                # y, x = np.mgrid[step/2:frame.shape[0]:step, step/2:frame.shape[1]:step].reshape(2,-1).astype(int)
                # p = np.dstack((x, y))
                # p.shape = -1, 1, 2
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
    def run_file(self):
        for img_file in os.listdir('.'):
            scale = 2
            frame = cv2.imread(img_file)
            frame = cv2.resize(frame, (frame.shape[1] / scale, frame.shape[0] / scale))

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if(self.frame_idx == 0):
                self.frame_idx += 1
                self.prev_gray = frame_gray
                continue

            step = 8
            y, x = np.mgrid[step/2:frame.shape[0]:step, step/2:frame.shape[1]:step].reshape(2,-1).astype(int)
            p = np.dstack((x, y))
            p.shape = -1, 1, 2
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

            test_name= img_file.split('.')[0]
            file_name = test_name + '_p0' + '.npy'
            np.save(file_name, p0)
            file_name = test_name + '_p1' + '.npy'
            np.save(file_name, p1)
            file_name = test_name + '_p0r' + '.npy'
            np.save(file_name, p0r)

            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            self.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            self.tracks = []

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 'D:/TEST_DATA/test_flow/data'
        # video_src = 'H:/2017-03-04-09-15-20/hiv00003.mp4'

    print(__doc__)
    os.chdir(video_src)
    App(video_src).run_file()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
