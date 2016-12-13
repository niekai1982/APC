import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from video import create_capture

save_flag = False
cv2.namedWindow('src_v', 1)
cap = create_capture('test.avi')

fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter('other.avi', cv2.cv.CV_FOURCC('I','4','2','0'),fps, size)
print videoWriter.isOpened()
frame_idx = 0
while True:
    _,frame = cap.read()
    cv2.imshow('src_v', frame)
    ch = 0xFF & cv2.waitKey(1)
    if save_flag:
        cv2.imwrite('1/'+str(frame_idx)+'.jpg', frame)
    if ch == 27:
        print 'end'
        break
    if ch == ord('s'):
        save_flag = True
    frame_idx += 1

cap.release()
videoWriter.release()
cv2.destroyAllWindows()


