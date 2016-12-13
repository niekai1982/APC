import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from video import create_capture


def WriteFrame(file_name, save_folder):

	save_path = save_folder 

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	cap = cv2.VideoCapture(file_name)	

	frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_rate   = cap.get(cv2.CAP_PROP_FPS)
	frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	for idx in range(frame_count):
		_,frame = cap.read()
		if not idx % 25:
			save_name = os.path.join(save_path, os.path.splitext(file_name)[0] + '_' + str(idx) + '.jpg')
			cv2.imwrite(save_name, frame)
			print file_name
			print str(idx)


if __name__ == '__main__':

	video_files = []
	os.chdir(r'E:/SAMPLE_DATA')
	sub_list = os.listdir('.')

	video_files = [elem for elem in os.listdir('.') if os.path.isfile(elem)]
	for file in video_files:
		WriteFrame(file, 'test')
