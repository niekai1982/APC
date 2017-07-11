# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:50:29 2017

@author: nieka
"""

import os
import re
import cv2
import matplotlib.pyplot as plt


def sampleWins(file_name):
    wins = []
    with open(file_name) as fo:
        out = fo.read()
    if 'heads' in out:
        out = out.splitlines()
        heads_coor = out[1:]
    else:
        return []
    for i in range(len(heads_coor)):
        rect = [int(elem) for elem in heads_coor[i].split()[1:5]]
        wins.append(rect)
    return wins


if __name__ == '__main__':
    cor_file_name = 'D:/data_sample/hiv00000_19196.jpg.txt'
    img_file_name = 'D:/data_sample/hiv00000_19196.jpg'

    scale = 2

    wins_rect = sampleWins(cor_file_name)

    img = cv2.imread(img_file_name)
    h, w = img.shape[:2]

    img_re = cv2.resize(img, (int(w * scale), int(h * scale)))
    for rect in wins_rect:
        pt1 = (int(rect[0] * scale), int(rect[1] * scale))
        pt2 = (int((rect[0] + rect[2]) * scale), int((rect[1] + rect[3]) * scale))
        cv2.rectangle(img_re, pt1, pt2, (0, 0, 255), 2)
    plt.imshow(img_re[:,:,::-1])
    plt.show()