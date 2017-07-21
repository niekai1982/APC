import numpy as np
import cv2
import video
import os
import matplotlib.pyplot as plt
from time import time
import cPickle
import skimage.io as siio



if __name__ == '__main__':
    files_path = './data/data'
    files = os.listdir(files_path)
    for file in files:
        file_name = os.path.join(files_path, file)
        img0 = siio.imread()
        plt.imshow(img0)
        plt.show()
