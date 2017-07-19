#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
import os
import matplotlib.pyplot as plt
from time import time
import cPickle
import matplotlib.pyplot as plt
from getFeature_test import gradient_Mag


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_motion(img_src, img_re, flow, scale, step=32):
    h, w = img_re.shape[:2]

    Gx, Gy = flow[:,:,0], flow[:,:,1]
    M, O = cv2.cartToPolar(Gx, Gy, angleInDegrees=False)
    O[O > np.pi] -= np.pi

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    vis = img_src.copy()
    for (i, j) in zip(y, x):
        val = M[i:(i + step), j:(j + step)].sum()
        if val > 2 * step * step:
            vis[i*scale:(i+step)*scale, j*scale:(j+step)*scale,2] = 255
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print(__doc__)

    file_path = 'H:/2017-03-04-09-15-20/hiv00003.mp4'
    # os.chdir(file_path)
    #
    # img_files = os.listdir('.')
    #
    scale = 4

    cap = cv2.VideoCapture(file_path)
    _, prev = cap.read()
    # prev = cv2.imread(img_files[0])
    prev = cv2.resize(prev, (prev.shape[1] / scale, prev.shape[0] / scale))
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    i = 2
    while True:
    # while i  < len(img_files):
        _, img = cap.read()
        _, img = cap.read()
        _, img = cap.read()
    #     _, img = cv2.imread(img_files[i])
        # i += 2
        img_r = cv2.resize(img, (img.shape[1] / scale, img.shape[0] / scale))

        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        start = time()
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        end = time()
        print(end - start)
        prevgray = gray
        previmg = img

        vis = draw_motion(img, img_r, flow, scale)

        cv2.imshow('flow', vis)
        # cv2.imshow('flow', draw_flow(gray, flow))
        # gx = flow[:,:,0]
        # gy = flow[:,:,1]
        # grident = gx * gx + gy * gy
        # out = (grident > 10) * 255
        # out = out.astype(np.uint8)
        # plt.imshow(out)
        # plt.colorbar()
        # plt.show()
        # out = out.astype(np.uint8)
        # cv2.imshow('test', out)
        # plt.imshow(flow[:,:,0] * flow[:,:,0] + flow[:,:,1] * flow[:,:,1])
        # plt.show()
        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
        if ch == ord('s'):
            f_name =  'flow' + '_' + str(i) + '.pkl'
            with open(f_name, 'wb') as fp:
                cPickle.dump(flow, fp)
                fp.close()
            f_name =  'previmg' + '_' + str(i) + '.jpg'
            cv2.imwrite(f_name, previmg)
            f_name =  'img' + '_' + str(i) + '.jpg'
            cv2.imwrite(f_name, img)
            print('save flow success')
    cv2.destroyAllWindows()
