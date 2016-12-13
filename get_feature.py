__author__ = 'kai'

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
from misc.constants import *
import feature.stub as stub
import feature.vector as vector
import misc.gradient as grad
from time import time

nor_size = (48, 32)
EXT_DICT = ['.jpg', 'bmp', '.png']

def get_files_name(path, ext_dict):
    files_ret = []
    files = os.listdir(sample_path)
    for file in files:
        if os.path.splitext(file)[-1] in ext_dict:
            files_ret.append(os.path.join(path, file))
    return files_ret

def get_icf_feature(img, vector_stub):
    data = cv2.resize(img, nor_size)
    start = time()
    chan = grad.get_channels(data)
    print 'get channel spend time: %f' % (time()-start)
    start = time()
    int_chan = grad.get_integral_channels(chan)
    print 'get integral channel spend time: %f' % (time()-start)
    start = time()
    feats = vector.extract(int_chan, vector_stub)
    print 'get feats vector spend time: %f' % (time()-start)
    return np.array(feats)


def get_hog_feature(img, vis_flag=0):
    data = cv2.resize(img, nor_size)[...,0]
    feature, hog_img = hog(data, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)
    if vis_flag:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(data, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_img)

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()

    return feature


def get_train_data(sample_path, vector_stub):

    os.chdir(sample_path)

    data = []
    label = []

    target_set = set([elem for elem in os.listdir(sample_path) if os.path.isdir(elem)])
    print target_set
    class_num = len(target_set)
    class_dict = dict(zip(target_set, range(class_num)))

    for root, dirs, files in os.walk(sample_path, topdown=False):
        word_data = [get_icf_feature(cv2.imread(os.path.join(root, name)), vector_stub) for name in files
                     if os.path.splitext(name)[-1] in EXT_DICT]
        for elem in word_data:
            label.append(class_dict[root.split(os.sep)[-1]])
            data.append(elem)

    data = np.array(data)
    label = np.array(label)
    return data, label, class_dict


if __name__ == '__main__':
    # img = cv2.imread('test.jpg')
    sample_path = r'e:\APC\sample_test'
    vector_stub_path = r'e:\APC\feature\canditate.vec'
    vec_stu = stub.read(vector_stub_path)
    # get_icf_feature(img, vec_stu)
    data, target, class_dict = get_train_data(sample_path, vec_stu)
    np.savez('data_test', data=data, target=target)
    dict_output = open('class_test.pkl', 'wb')
    pickle.dump(class_dict, dict_output)
    dict_output.close()

