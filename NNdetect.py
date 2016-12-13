__author__ = 'kai'
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from common import mosaic
from BrainTest import load_model,predict
from get_feature import get_hog_feature
from time import time
from get_feature import get_hog_feature, get_icf_feature
import feature.stub as stub


def detect(img, model, rect_init_size, scale_set, step_size, vec_stu):
    rect_cor = sliding_window(img.shape[:-1], rect_init_size, scale_set, step_size)
    print 'sample to classify:%d' % len(rect_cor)
    res_prob = []
    head_n = []
    head_p = []
    for rect in rect_cor:
        x_io, y_io, w, h = rect
        sample = img[y_io:y_io+h, x_io:x_io+w]
        start = time()
        sample_feature = get_icf_feature(sample, vec_stu)
        end = time()
        print 'extract features spend time:%f' % (end-start)
        probs, res = predict(model, sample_feature)
        end_1 = time()
        print 'sample classify spend time:%f' % (end_1-end)
        if not res and probs[res]:
            head_p.append(cv2.resize(sample,(64,64)))
        else:
            head_n.append(cv2.resize(sample, (64,64)))
    return head_p,head_n


def sliding_window(img_size, rect_init_size, scale_set, step_size):
    h, w = img_size
    rect_cor = []
    for scale in scale_set:
        w_range = np.arange(0, w - int(rect_init_size[0]*scale), step_size[0])
        h_range = np.arange(0, h - int(rect_init_size[1]*scale), step_size[1])
        x_grid, y_grid = np.meshgrid(w_range, h_range)
        for idx_x in range(x_grid.shape[0]):
            for idx_y in range(x_grid.shape[1]):
                rect_cor.append([x_grid[idx_x, idx_y], y_grid[idx_x, idx_y], int(rect_init_size[0]*scale), int(rect_init_size[1] * scale)])
    return rect_cor


if __name__ == '__main__':
    print 'WORK PATH = ' + os.getcwd()
    model_path = r'E:\APC\sample_test\model.npy'
    vector_stub_path = r'e:\APC\feature\canditate.vec'

    vec_stu = stub.read(vector_stub_path)

    test_path = r'E:\APC\sample'
    test_files = os.listdir(test_path)
    for test_file in test_files:
        img = plt.imread(os.path.join(test_path,test_file))[45:246,160:565,:]
        img = cv2.resize(img, (256, 128))
        # num_features = len(get_hog_feature(img))
        model = load_model(model_path, input_dim=5000, hidden_dim=200, out_dim=2)
        scale_set = np.arange(0.8, 1.2, 0.1)
        rect_init_size = [50, 30]
        step_size = [8, 8]
        head_p, head_n = detect(img, model, rect_init_size, scale_set, step_size, vec_stu)
        plt.subplot(1,2,1)
        plt.imshow(mosaic(10, head_p))
        plt.title('HEAD')
        plt.subplot(1,2,2)
        plt.imshow(mosaic(20, head_n))
        plt.title('NO HEAD')
        plt.show()
