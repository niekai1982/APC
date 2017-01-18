import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common import draw_str, RectSelector, draw_rect

roi_x = 160
roi_y = 32
roi_width = 400
roi_height = 240

rect_scale = np.arange(1, 2.4, 0.4)
rect_width_init = 64
rect_height_init = 48
rect_shift = 5

def split_window(sample_path, save_path):
    idx = 0
    sample_files = os.listdir(sample_path)

    for file in sample_files:
        file_name = os.path.join(sample_path, file)
        img = cv2.imread(file_name)[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        h, w = img.shape[:-1]
        for scale in rect_scale:
            rect_width = int(rect_width_init * scale)
            rect_height = int(rect_height_init * scale)
            x = np.arange(0, w-rect_width, rect_shift)
            y = np.arange(0, h-rect_height, rect_shift)
            rect_coor = np.meshgrid(x, y)
            for rect_x, rect_y in zip(rect_coor[0].flatten(), rect_coor[1].flatten()):
                sample = img[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]
                sample = cv2.resize(sample, (32, 32))
                file_name = os.path.join(save_path, str(idx) + '.jpg')
                cv2.imwrite(file_name, sample)
                idx += 1
                print idx


def get_manul(sample_path, save_path):

    sample_rects = []
    file_idx = 0
    out_flag = False

    def Onrect(rect):
        sample_rects.append(rect)

    def filter_extr(file_name):
        return os.path.splitext(file_name)[-1] in ['.JPG','.jpg','.bmp']

    cv2.namedWindow('Frame', 1)
    rect_sel = RectSelector('Frame', Onrect)

    sample_files = os.listdir(sample_path)
    sample_files = filter(filter_extr, sample_files)

    for file in sample_files:

        sample_rects = []
        frame = cv2.imread(os.path.join(sample_path, file))
        frame = frame[:,:,::-1]
        file_name = os.path.splitext(file)[0]


        while True:
            vis = frame.copy()
            rect_sel.draw(vis)
            draw_str(vis, (20,20), file_name)
            for rect in sample_rects:
                x = rect[0]
                y = rect[1]
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                draw_str(vis, (rect[0],rect[1] - 5), '(%d,%d,%d,%d)' % (x,y,w,h))
                draw_rect(vis, rect)
            cv2.imshow('Frame', vis)
            ch = cv2.waitKey(1)
            if ch == 27:
                cv2.destroyAllWindows()
                out_flag = True
                break
            if ch == ord('n'):
                sample_rects = []
                break
            if ch == ord('s'):
                num_rects = len(sample_rects)
                print num_rects
                coor_file_name = file_name + '.txt'
                coor_file_path = os.path.join(sample_path, coor_file_name)
                fp = open(coor_file_path, 'wb')
                for idx_rect, rect in enumerate(sample_rects):
                    x0, y0, x1, y1 = rect
                    x_c = (x0 + x1) * 1. / 2
                    y_c = (y0 + y1) * 1. / 2
                    w = (x1 - x0) * 1.
                    h = (y1 - y0) * 1.
                    coor_res = '%f %f %f %f' % (x_c / vis.shape[1], y_c / vis.shape[0], w / vis.shape[1], h / vis.shape[0])
                    fp.write("0" + " " + coor_res + "\n")
                fp.close()
            if ch == ord('r'):
                sample_rects = []
        if out_flag:
            break


if __name__ == '__main__':
    get_manul(r'E:\car_data', r'E:\PROGRAM\APC\sample_test\8\sample')

    pass
