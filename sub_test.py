# from __future__ import print_function

import numpy as np
import cv2
import video
import os
import matplotlib.pyplot as plt
from time import time
import cPickle
from skimage import io


def frameSub(img1, img2, img3, thr=20):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    # gray1 = img1[:,:,0]
    # gray2 = img2[:,:,0]
    # gray3 = img3[:,:,0]

    gray1, gray2, gray3 = [cv2.resize(elem, (320, 160)) for elem in [gray1, gray2, gray3]]
    gray1, gray2, gray3 = [cv2.medianBlur(elem, 3) for elem in [gray1, gray2, gray3]]

    d1 = gray2 - gray1
    d2 = gray3 - gray2

    b1 = np.where(d1>thr, 1, 0)
    b2 = np.where(d2>thr, 1, 0)

    B = b1 * b2

    # B = d1 * d2

    return B


def draw_motion(img, motion, step=32):
    h, w = img.shape[:2]

    # Gx, Gy = flow[:,:,0], flow[:,:,1]
    # M, O = cv2.cartToPolar(Gx, Gy, angleInDegrees=False)
    # O[O > np.pi] -= np.pi

    y, x = np.mgrid[0:h:step, 0:w:step].reshape(2,-1).astype(int)
    vis = img.copy()
    for (i, j) in zip(y, x):
        val = motion[i:(i + step), j:(j + step)].sum()
        if val > 0.2 * step * step:
            vis[i:(i+step), j:(j+step),0] = 255
    return vis


def video_test(video_src):
    cam = video.create_capture(video_src)

    while True:
        ret, img1 = cam.read()
        ret, img2 = cam.read()
        ret, img3 = cam.read()

        start = time()
        B = frameSub(img1, img2, img3, thr=10)
        end = time()
        print end - start

        vis = draw_motion(img2, B, step=16)
        # vis = B * 255
        # vis = vis.astype(np.uint8)

        cv2.imshow('flow', vis)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break



if __name__ == '__main__':
    import sys
    print(__doc__)

    video_src = 'H:/2017-03-04-09-15-20/hiv00003.mp4'
    video_test(video_src)


    # img1 = io.imread('./data/data/1277.jpg')
    # img2 = io.imread('./data/data/1278.jpg')
    # img3 = io.imread('./data/data/1279.jpg')
    #
    # B = frameSub(img1, img2, img3)
    # vis = draw_motion(img2, B)
    #
    # plt.imshow(vis)
    # plt.show()



    # file_path = 'D:/TEST_DATA/test_flow/data'
    # os.chdir(file_path)
    #
    # img_files = os.listdir('.')
    #
    # scale = 1
    #
    # prev = cv2.imread(img_files[0])
    # prev = cv2.resize(prev, (prev.shape[1] / scale, prev.shape[0] / scale))
    # prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # show_hsv = False
    # show_glitch = False
    # cur_glitch = prev.copy()
    #
    # i = 4
    # while i < len(img_files):
    #     img = cv2.imread(img_files[i])
    #     img = cv2.resize(img, (img.shape[1] / scale, img.shape[0] / scale))
    #
    #     i += 4
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (3,3), 3)
    #
    #     # start = time()
    #     # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     # end = time()
    #     # print(end - start)
    #
    #     out = np.abs(gray - prevgray)
    #     out[out > 250] = 0
    #     out[out < 50] = 0
    #     # plt.imshow(out)
    #     # plt.colorbar()
    #     # plt.show()
    #     # plt.imshow(out)
    #     # plt.show()
    #
    #     # plt.imshow(out)
    #     # plt.show()
    #
    #     cv2.imshow('flow', out)
    #     #
    #     ch = 0xFF & cv2.waitKey(1000)
    #     if ch == 27:
    #         break
    #     # if ch == ord('1'):
    #     #     show_hsv = not show_hsv
    #     #     print('HSV flow visualization is', ['off', 'on'][show_hsv])
    #     # if ch == ord('2'):
    #     #     show_glitch = not show_glitch
    #     #     if show_glitch:
    #     #         cur_glitch = img.copy()
    #     #     print('glitch is', ['off', 'on'][show_glitch])
    #     prevgray = gray
    # cv2.destroyAllWindows()
