import imutils
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import pyramid_gaussian
import time


def pyramid(image, scale=1.2, minSize=(32,64)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0] - windowSize[1], stepSize):
        for x in xrange(0, image.shape[1] - windowSize[0], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':

#    parser = argparse.ArgumentParser()
#    parser.add_argument("-i", "--image",  default='test.jpg' ,help="Path to the image")
#    parser.add_argument("-s", "--scale", type=float, default=1.2, help="scale factor size")
#    args = vars(parser.parse_args())
#
#    image = cv2.imread(args["image"])
#    (winH, winW) = (32, 32)
#
#
#
#    start = time.time()
#    numIter = 0
#    for resized in pyramid(image, scale=args["scale"]):
#        for(x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
#            if window.shape[0] != winH or window.shape[1] != winW:
#                continue
#            numIter += 1
#            # clone = resized.copy()
#            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
#            # cv2.imshow("window", clone)
#            # cv2.waitKey(1)
#            # time.sleep(0.01)
#    end = time.time()
#
#    print numIter
#    print (end - start)
#
#    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    cv2.equalizeHist(grayscale, grayscale)
#    cascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')
#
#    start = time.time()
#    rects = cascade.detectMultiScale(grayscale, scaleFactor=1.01, minNeighbors=1, minSize=(16, 16))
#    end = time.time()
#    print (end - start)
#    clone = image.copy()
#
#    for rect in rects:
#        print rect
#        x, y = rect[0], rect[1]
#        w, h = rect[2], rect[3]
#        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#    plt.imshow(clone)
#    plt.show()
    
    image = cv2.imread('test.jpg')
    grayscale = image[:,:,1]
    # grayscale = cv2.resize(grayscale, (grayscale.shape[1] / 2, grayscale.shape[0] / 2))
    # plt.imshow(grayscale)
    

    hog = cv2.HOGDescriptor(_winSize=(32,32),_blockSize=(16,16),_blockStride=(8,8),_cellSize=(8,8),_nbins=9)
    win_nfeature = hog.getDescriptorSize()
#    grayscale = cv2.resize(grayscale, (32,32))
#    
#    start = time.time()
#    for i in range(10000):
#        desc = hog.compute(grayscale)
#    end = time.time()
#    print "hog spend time : %f" % (end - start)
    start = time.time()
    hog_feature = hog.compute(grayscale[:,:32], winStride=(8,8))
    end = time.time()
    print (end - start)
    hog_feature_s = []
    start = time.time()
    idx = 0
    for (x, y, window) in sliding_window(grayscale[:,:32], stepSize=8, windowSize=(32,32)):
        desc = hog.compute(window)
#        plt.plot(desc)
#        plt.plot(hog_feature[idx * win_nfeature : (idx + 1) * win_nfeature])
#        plt.show()
        idx += 1
        print idx
    end = time.time()
    print (end - start)
#    plt.imshow(grayscale)
    
#    start = time.time()
#    temp_a = hog.compute(grayscale)
#    temp_b = hog.compute(grayscale[:32,:32])
#    end = time.time()
#    print (end - start)
#    start = time.time()
#    for (i, resized) in enumerate(pyramid(grayscale)):
#        print i
#        temp = hog.compute(resized)
#    end = time.time()
#    print (end - start)


# # used imutils
# for (i, resized) in enumerate(pyramid(image, scale=args["scale"])):
#     plt.imshow(resized)
#     plt.show()
#
# # used skimage.pyramid_gaussian
# for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
#     if resized.shape[0] < 30 or resized.shape[1] < 30:
#         break
#     plt.imshow(resized)
#     plt.show()
