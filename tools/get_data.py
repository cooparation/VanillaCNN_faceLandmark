# coding: utf-8

import os
from os.path import exists, join, isfile
import numpy as np
from utils import BBox 

def getDataFromTXT(filepath, test=False):
	'''
	Get data from dataset mentioned in paper.
	Input:
	- filepath: trainImageList or testImageList
	Output:
	- A tuple of (imgpath, bbox, landmark)
		- imgpath: train image or test image
		- bbox: type of BBox
		- landmark: (5L, 2L) of [0,1]
	'''
	dirname = os.path.dirname(filepath)
	f = open(filepath, 'r')
	data = []

	for line in f.readlines():
		s = line.strip().split(' ')
		imgPath = os.path.join(dirname, s[0].replace('\\', '/'))
		bbox = map(int, [s[1], s[2], s[3], s[4]])
		bbox = BBox(bbox)

		if test:
			x_max=bbox.right#bbox[1]
                        x_min=bbox.left#bbox[0] 
                        y_max=bbox.bottom#bbox[3]
                        y_min=bbox.top#bbox[2]
                        # enlarge bounding box
                        w, h = x_max-x_min, y_max-y_min
                        w = h = min(w, h)
                        ratio = 0
                        x_new = x_min - w*ratio
                        y_new = y_min - h*ratio
                        w_new = w*(1 + 2*ratio)
                        h_new = h*(1 + 2*ratio)
                        new_bbox = map(int, [x_new, x_new+w_new, y_new, y_new+h_new])
                        new_bbox = BBox(new_bbox)
			data.append((imgPath, new_bbox))
			continue
		landmark = np.zeros((5,2))
		for i in range(0,5):
			landmark[i] = (float(s[5+i*2]), float(s[5+i*2+1]))
		landmark = bbox.projectLandmark(landmark) #[0,1]
		#print landmark
		fit=0
		for i in range(0,5):
			if landmark[i,0]<0 or landmark[i,0]>1 or landmark[i,1]<0 or landmark[i,1]>1:
				fit=1
				break
		if fit==0:
			data.append((imgPath, bbox, landmark))
	return data

def getDataFromTXT_68(filepath, test=False):
	'''
	Get data from dataset mentioned in paper.
	Input:
	- filepath: trainImageList or testImageList
	Output:
	- A tuple of (imgpath, bbox, landmark)
		- imgpath: train image or test image
		- bbox: type of BBox
		- landmark: (5L, 2L) of [0,1]
	'''
	dirname = os.path.dirname(filepath)
	f = open(filepath, 'r')
	data = []

	for line in f.readlines():
		s = line.strip().split(' ')
		imgPath = os.path.join(dirname, s[0].replace('\\', '/'))
		bbox = map(int, [s[1], s[2], s[3], s[4]])
		bbox = BBox(bbox)

		if test:
			data.append((imgPath, bbox))
			continue
		landmark = np.zeros((68,2))
		for i in range(0,68):
			landmark[i] = (float(s[5+i*2]), float(s[5+i*2+1]))
		landmark = bbox.projectLandmark(landmark) #[0,1]
		data.append((imgPath, bbox, landmark))
	return data

def getDataFromTXT_68_scale(filepath, test=False):
	'''
	Get data from dataset mentioned in paper.
	Input:
	- filepath: trainImageList or testImageList
	Output:
	- A tuple of (imgpath, bbox, landmark)
		- imgpath: train image or test image
		- bbox: type of BBox
		- landmark: (5L, 2L) of [0,1]
	'''
	dirname = os.path.dirname(filepath)
	f = open(filepath, 'r')
	data = []

	for line in f.readlines():
		s = line.strip().split(' ')
		imgPath = os.path.join(dirname, s[0].replace('\\', '/'))
		bbox = map(int, [s[1], s[2], s[3], s[4]])
		bbox = BBox(bbox)

		if test:
			x_max=bbox.right#bbox[1]
                        x_min=bbox.left#bbox[0] 
                        y_max=bbox.bottom#bbox[3]
                        y_min=bbox.top#bbox[2]
                        # enlarge bounding box
                        w, h = x_max-x_min, y_max-y_min
                        w = h = min(w, h)
                        ratio = 0.2
                        x_new = x_min - w*ratio
                        y_new = y_min - h*ratio
                        w_new = w*(1 + 2*ratio)
                        h_new = h*(1 + 2*ratio)
                        new_bbox = map(int, [x_new, x_new+w_new, y_new, y_new+h_new])
                        new_bbox = BBox(new_bbox)
			data.append((imgPath, new_bbox))
			continue
		landmark = np.zeros((68,2))
		for i in range(0,68):
			landmark[i] = (float(s[5+i*2]), float(s[5+i*2+1]))
		landmark = bbox.projectLandmark(landmark) #[0,1]
		fit=0
		for i in range(0,68):
			if landmark[i,0]<0 or landmark[i,0]>1 or landmark[i,1]<0 or landmark[i,1]>1:
				fit=1
				break
		if fit==0:
			data.append((imgPath, bbox, landmark))
	return data




def getDataFromTXT_5(filepath, test=False):
	'''
	Get data from dataset mentioned in paper.
	Input:
	- filepath: trainImageList or testImageList
	Output:
	- A tuple of (imgpath, bbox, landmark)
		- imgpath: train image or test image
		- bbox: type of BBox
		- landmark: (5L, 2L) of [0,1]
	'''
	dirname = os.path.dirname(filepath)
	f = open(filepath, 'r')
	data = []

	for line in f.readlines():
		s = line.strip().split(' ')
		imgPath = os.path.join(dirname, s[0].replace('\\', '/'))
		bbox = map(int, [s[1], s[2], s[3], s[4]])
		bbox = BBox(bbox)

		if test:
			data.append((imgPath, bbox))
			continue
		landmark = np.zeros((5,2))
		index_list = [36,45,33,48,54]
		for i in range(0,5):
			landmark[i] = (float(s[5+index_list[i]*2]), float(s[5+index_list[i]*2+1]))
		landmark = bbox.projectLandmark(landmark) #[0,1]
		#print landmark
		fit=0
		for i in range(0,5):
			if landmark[i,0]<0 or landmark[i,0]>1 or landmark[i,1]<0 or landmark[i,1]>1:
				fit=1
				break
		if fit==0:
			data.append((imgPath, bbox, landmark))
	return data



def load_celeba_data():
	'''
	load celeba dataset and crop the face box
	Return a tuple of:
		- img_path: dataset/celeba/000001.jpg
		- bbox: object of BBox
		- landmark: (5L, 2L) of [0,1]
	'''
	text = '/home/cunjian/code/caffe/examples/dataset/celeba/list_landmarks_celeba.txt'
	# text = 'E:\\dataset\\CelebA\\list_landmarks_celeba.txt'
	fin = open(text, 'r')
	n = int(fin.readline().strip())
	fin.readline() # drop this line [lefteye_x, lefteye_y, ...]

	result = []
	for i in range(n):
		line = fin.readline().strip()
		components = line.split()
		img_path = join('../dataset/img_celeba', components[0])
		# img_path = join('E:\\dataset\\CelebA\\img_celeba', components[0])
		landmark = np.asarray([int(value) for value in components[1:]], dtype=np.float32)
		landmark = landmark.reshape(len(landmark) / 2, 2)

		# crop face box
		x_max, y_max = landmark.max(0)
		x_min, y_min = landmark.min(0)
		w, h = x_max-x_min, y_max-y_min
		w = h = min(w, h)
		ratio = 0 # default 0.5
		x_new = x_min - w*ratio
		y_new = y_min - h*ratio
		w_new = w*(1 + 2*ratio)
		h_new = h*(1 + 2*ratio)
		bbox = map(int, [x_new, x_new+w_new, y_new, y_new+h_new])
		bbox = BBox(bbox)

		# normalize landmark
		landmark = bbox.projectLandmark(landmark)

		#print landmark, if uncommented, the program will report error since some samples are omitted
		#fit=0
		#for index in range(0,5):
		#	if landmark[index,0]<0 or landmark[index,0]>1 or landmark[index,1]<0 or landmark[index,1]>1:
		#		fit=1
		#		break
		#if fit==0:
		#	result.append((img_path, bbox, landmark))

		result.append((img_path, bbox, landmark))

	fin.close()
	return result


def get_train_val_test_list():
	'''
	Get train list and validation and test list of celeba
	'''
	txt = os.path.join('/home/cunjian/code/caffe/examples/dataset/celeba', 'list_eval_partition.txt')
	f = open(txt, 'r')
	train_list = []
	val_list = []
	test_list = []

	for line in f.readlines():
		s = line.split()
		if s[1] == '0':
			train_list.append(s[0])
		elif s[1] == '1':
			val_list.append(s[0])
		elif s[1] == '2':
			test_list.append(s[0])
	train_list = [int(_.split('.')[0]) for _ in train_list]
	val_list = [int(_.split('.')[0]) for _ in val_list]
	test_list = [int(_.split('.')[0]) for _ in test_list]
	return train_list, val_list, test_list
