import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common import draw_str, RectSelector

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

    def Onrect(rect):
        sample_rects.append(rect)

    cv2.namedWindow('Frame', 1)
    rect_sel = RectSelector('Frame', Onrect)

    sample_files = os.listdir(sample_path)
    frame = cv2.imread(os.path.join(sample_path, sample_files[file_idx]))
    file_name = os.path.splitext(sample_files[file_idx])[0]
    file_idx += 1

    paused_flag = True

    while True:
        if not paused_flag and file_idx < len(sample_files):
            frame = cv2.imread(os.path.join(sample_path, sample_files[file_idx]))
            file_name = os.path.splitext(sample_files[file_idx])[0]
            file_idx += 1
            paused_flag = True

        vis = frame.copy()
        rect_sel.draw(vis)

        cv2.imshow('Frame', vis)
        ch = cv2.waitKey(1)
        if ch == 27:
            cv2.destroyAllWindows()
            print sample_rects
            break
        if ch == ord('n'):
            paused_flag = False
            continue
        if ch == ord('s'):
            num_rects = len(sample_rects)
            print num_rects
            for idx_rect, rect in enumerate(sample_rects):
                x0, y0, x1, y1 = rect
                rect_name = os.path.join(save_path, file_name + '_' + str(idx_rect) + '.jpg')
                cv2.imwrite(rect_name, frame[y0:y1, x0:x1])
            sample_rects = []
        if ch == ord('r'):
            pass
    return


if __name__ == '__main__':
    get_manul('./sample', './test')
    pass
