#coding=utf-8
import os
import sys
import numpy as np
import cv2
import math
import signal
import random
import time
from multiprocessing import Process, Queue, Event

from landmark_augment import LandmarkAugment
from landmark_helper import LandmarkHelper

exitEvent = Event() # for noitfy all process exit.

#def handler(sig_num, stack_frame):
#    global exitEvent
#    exitEvent.set()
#signal.signal(signal.SIGINT, handler)

class BatchReader():
    def __init__(self, **kwargs):
        # param
        self._kwargs = kwargs
        self._need_augment = kwargs['need_augment']
        self._batch_size = kwargs['batch_size']
        self._process_num = kwargs['process_num']
        # total lsit
        self._sample_list = [] # each item: (filepath, landmarks, ...)
        self._total_sample = 0
        # real time buffer
        self._process_list = []
        self._output_queue = []
        for i in range(self._process_num):
            self._output_queue.append(Queue(maxsize=3)) # for each process
        # epoch
        self._idx_in_epoch = 0
        self._curr_epoch = 0
        self._max_epoch = kwargs['max_epoch']
        # start buffering
        self._start_buffering(kwargs['input_paths'], kwargs['landmark_type'], kwargs['has_bbox'])

    def batch_generator(self):
        __curr_queue = 0
        while True:
            self.__update_epoch()
            while True:
                __curr_queue += 1
                if __curr_queue >= self._process_num:
                    __curr_queue = 0
                try:
                    image_list, landmarks_list = self._output_queue[__curr_queue].get(block=True, timeout=0.01)
                    break
                except Exception as ex:
                    pass
            yield image_list, landmarks_list

    def get_epoch(self):
        return self._curr_epoch

    def should_stop(self):
        if exitEvent.is_set() or self._curr_epoch > self._max_epoch:
            exitEvent.set()
            self.__clear_and_exit()
            return True
        else:
            return False

    def __clear_and_exit(self):
        print ("[Exiting] Clear all queue.")
        while True:
            time.sleep(1)
            _alive = False
            for i in range(self._process_num):
                try:
                    self._output_queue[i].get(block=True, timeout=0.01)
                    _alive = True
                except Exception as ex:
                    pass
            if _alive == False: break
        print ("[Exiting] Confirm all process is exited.")
        for i in range(self._process_num):
            if self._process_list[i].is_alive():
                print ("[Exiting] Force to terminate process %d"%(i))
                self._process_list[i].terminate()
        print ("[Exiting] Batch reader clear done!")

    def _start_buffering(self, input_paths, landmark_type, has_bbox):
        if type(input_paths) in [str, unicode]:
            input_paths = [input_paths]
        for input_path in input_paths:
            for line in open(input_path):
                info = LandmarkHelper.parse(line, landmark_type, has_bbox)
                self._sample_list.append(info)
        self._total_sample = len(self._sample_list)
        num_per_process = int(math.ceil(self._total_sample / float(self._process_num)))

        # multi-thread to process
        for idx, offset in enumerate(range(0, self._total_sample, num_per_process)):
            p = Process(target=self._process, args=(idx, self._sample_list[offset: offset+num_per_process]))
            p.start()
            self._process_list.append(p)

    def preProcessImage(self, img):
        """
        preprocess: image normalization
        """
        img = img.astype(np.float32)
        m = img.mean()
        s = img.std()
        img = (img - m) /(1.0e-6 +s)
        return img

    def _process(self, idx, sample_list):
        __landmark_augment = LandmarkAugment()
        # read all image to memory to speed up!
        if self._kwargs['buffer2memory']:
            print ("[Process %d] Start to read image to memory! Count=%d"%(idx, len(sample_list)))
            scale_value = 2.2 #3.5
            sample_list = __landmark_augment.mini_crop_by_landmarks(sample_list, scale_value, self._kwargs['img_format'])
            print ("[Process %d] Read all image to memory finish!"%(idx))
        sample_cnt = 0 # count for one batch
        image_list, landmarks_list = [], [] # one batch list
        while True:
            for sample in sample_list:
                # preprocess
                if type(sample[0]) in [str, unicode]: #read jpg image
                    image = cv2.imread(sample[0])
                   # if self._kwargs['img_format'] == 'RGB':
                   #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                   # if self._kwargs['img_format'] == 'GRAY':
                   #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif self._kwargs['img_format'] == 'GRAY':
                    image = cv2.imdecode(sample[0], cv2.IMAGE_GRAYSCALE ) #read jpg stream image
                else:
                    image = cv2.imdecode(sample[0], cv2.IMREAD_COLOR) #read jpg stream image
                    #image = cv2.imdecode(sample[0], cv2.CV_LOAD_IMAGE_COLOR)
                landmarks = sample[1].copy()# keep deep copy

                if self._need_augment:
                    #scale_range = (2.7, 3.3)
                    scale_range = (1.5, 2.3)
                    image_new, landmarks_new = __landmark_augment.augment(image, landmarks, self._kwargs['img_size'],
                                                self._kwargs['max_angle'], scale_range)
                    #cv2.imwrite("./output_tmp/tmp%d.jpg"%(sample_cnt), image_new)
                else:
                    ori_h, ori_w = image.shape[:2]
                    # make landmark -> [0,1]
                    landmarks = (1.0*landmarks) / (ori_w, ori_h)
                    landmarks_new = landmarks.flatten()
                    new_w = self._kwargs['img_size']
                    new_h = self._kwargs['img_size']
                    image_new = cv2.resize(image, (new_w, new_h))

                #cv2.imwrite('./testing/test_images/test40x40.jpg', image_new)

                # caffe data format whc->chw
                im_ = np.transpose(image_new, (2, 0, 1))
                self.preProcessImage(im_)
                #im_ = im_.astype(np.float32)
                #im_ = im_/127.5-1.0

                # sent a batch -- check image
                #image_list.append(image_new) # open to check the image
                image_list.append(im_)

                sample_cnt += 1
                landmarks_list.append(landmarks_new)
                if sample_cnt >= self._kwargs['batch_size']:
                    self._output_queue[idx].put((np.array(image_list), np.array(landmarks_list)))
                    sample_cnt = 0
                    image_list, landmarks_list = [], []
                # if exit
                if exitEvent.is_set():
                    break
            if exitEvent.is_set():
                break
            np.random.shuffle(sample_list)

    def __update_epoch(self):
        self._idx_in_epoch += self._batch_size
        if self._idx_in_epoch > self._total_sample:
            self._curr_epoch += 1
            self._idx_in_epoch = 0

# use for unit test
if __name__ == '__main__':
    kwargs = {
        #'input_paths': "data/try.txt",
        'input_paths': "data/trainImageList.txt",
        'need_augment': True,
        'landmark_type': 5,
        'has_bbox':True,
        #'batch_size': 10,
        'batch_size': 512,
        'process_num': 30,
        'img_format': 'RGB',
        'img_size': 40, #112,
        'max_angle': 5, #10,
        'buffer2memory': True,
        'max_epoch': 100,
    }
    b = BatchReader(**kwargs)
    g = b.batch_generator()
    output_folder = "output_tmp"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    import time
    start_time = time.time()
    #for i in range(1000000):
    for i in range(19):
        end_time = time.time()
        print ("get new batch...step: %d. epoch: %d. cost: %.3f"%(i, b.get_epoch(), end_time-start_time))
        start_time = end_time
        batch_image, batch_landmarks = g.next()
        print 'out batch_image', len(batch_image)

        for idx, (image, landmarks) in enumerate(zip(batch_image, batch_landmarks)):
           #if idx > 20: # only see first 10
           #    break
           landmarks = landmarks.reshape([-1, 2])
           for l in landmarks:
               print 'get landmark', l
               ii = tuple(l * (kwargs['img_size'], kwargs['img_size']))
               cv2.circle(image, (int(ii[0]), int(ii[1])), 2, (0,255,0), -1)
           print "image channels", image.shape[0], image.shape[1], image.shape[2]
           cv2.imwrite("%s/%d.jpg"%(output_folder, idx), image)
           print ("write to %s/%d.jpg"%(output_folder, idx))
    print ("Done...Press ctrl+c to exit me")
    sys.exit()

