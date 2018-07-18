# by Bob in 20170326
import caffe
import numpy as np
import cv2
import os
import string, random

from batch_reader import BatchReader

class ImageInputDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        self.params = eval(self.param_str)

        # store input as class variables
        self.batch_size = self.params['batch_size']

        # store landmark_type
        self.num_points = self.params['landmark_type']

        # store data channels
        if self.params['img_format'] == 'RGB':
            self.num_channels = 3
        elif self.params['img_format'] == 'GRAY':
            self.num_channels = 1
        else:
            raise Exception("Unsupport img_format ...")

        # Create a batch loader to load the images.
        # we can disable reader when test
        if self.params['need_reader']:
            self.batch_reader = BatchReader(**self.params)
            self.batch_generator = self.batch_reader.batch_generator()

        # === reshape tops ===
        top[0].reshape(
            self.batch_size, self.num_channels, self.params['img_size'], self.params['img_size'])
        top[1].reshape(
            self.batch_size, self.num_points* 2)

    def preProcessImage(self, imgs):
        """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
        """
        imgs = imgs.astype(np.float32)
        for i, img in enumerate(imgs):
            m = img.mean()
            s = img.std()
            imgs[i] = (img - m) / s
        return imgs

    def forward(self, bottom, top):
        """
        Load data.
        """
        images, labels = self.batch_generator.next()
        #print 'liusanjun images num', len(images)
        self.preProcessImage(images)
        top[0].data[...] = images
        top[1].data[...] = labels

    def reshape(self, bottom, top):
        # === reshape tops ===
        top[0].reshape(
            self.batch_size, self.num_channels, self.params['img_size'], self.params['img_size'])
        top[1].reshape(
            self.batch_size, self.num_points* 2)

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


