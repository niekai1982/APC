___author__ = 'kai'

import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgbConvert(I, dstColorSpace):
    """
    Convert RGB image to other color spaces
    :param I : src image
    :param dstColorSpace: dst color space
    """
    colorSpaces = {'gray': cv2.COLOR_RGB2GRAY, 'luv': cv2.COLOR_RGB2LUV, 'hsv':cv2.COLOR_RGB2HSV}
    if dstColorSpace not in colorSpaces.keys():
        return I.astype(np.single) / 255
    else:
        return cv2.cvtColor(I, colorSpaces[dstColorSpace]).astype(np.single) / 255


if __name__ == '__main__':
    img = cv2.imread('peppers.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_hsv = rgbConvert(img, 'hsv')
    img_luv = rgbConvert(img, 'luv')
    img_gray = rgbConvert(img, 'gray')

    plt.subplot(1, 3, 1)
    plt.imshow(img_gray)
    plt.subplot(1, 3, 2)
    plt.imshow(img_hsv)
    plt.subplot(1, 3, 3)
    plt.imshow(img_luv)
    plt.show()