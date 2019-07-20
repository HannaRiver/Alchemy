#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
Classification with normal Caffe
Author: He Na
Time: 2018.12.29
'''
import os
import os.path as osp
import sys
import cv2
import numpy as np
import shutil
from alchemy_config import cfg
utils_dir = cfg.UTILS_DIR
caffe_root = cfg.CAFFE_ROOT.CLASSIFY
sys.path.insert(0, utils_dir)
from file2list import readTxt
from resize_img import undeform_center_resize, resize_with_center_pad, undeform_resize
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


class CaffeClassification:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, mean_value=[0, 0, 0], input_scale=1, transpose=(2, 0, 1), resize_type=''):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        image_resize_h, image_resize_w = image_resize
        self.resize_type = resize_type
        self.image_resize_h = image_resize_h
        self.image_resize_w = image_resize_w
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', transpose)
        self.transformer.set_mean('data', np.array(mean_value))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_input_scale('data', input_scale)
        self.transformer.set_channel_swap('data', (2, 1, 0))
    
    def classify(self, image_file, layername='sofmax'):
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        image = caffe.io.load_image(image_file) if type(image_file) == type('hello') else image_file / 255.0

        if self.resize_type == 'adaptiveBinary':
            image = cv2.imread(image_file) if type(image_file) == type('Hi') else image_file 
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gauss = cv2.GaussianBlur(gray, (3, 3), 1)
            gaus = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
            image = cv2.cvtColor(gaus, cv2.COLOR_GRAY2RGB)/255.0
        elif self.resize_type == 'undeform_resize':
            image = cv2.imread(image_file) if type(image_file) == type('Hi') else image_file 
            image = undeform_resize(image, self.image_resize_h, self.image_resize_w, self.mean_value)/255.0  
        elif self.resize_type == 'undeform_resize_with_white':
            image = cv2.imread(image_file) if type(image_file) == type('Hi') else image_file
            image = undeform_resize(image, self.image_resize_h, self.image_resize_w, [255, 255, 255])/255.0  

        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        self.net.forward()

        prob = self.net.blobs[layername].data[0].flatten()
        index_prob = prob.argsort()[-1]

        return prob, index_prob
    
    def get_some_layer(self, image_file, layernames=['sofmax']):
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        if type(image_file) == type('hello'):
            image = caffe.io.load_image(image_file)
        else:
            image = image_file / 255.0

        if self.resize_type == 'adaptiveBinary':
            if type(image_file) == type('hello'):
                image = cv2.imread(image_file)
            else:
                image = image_file
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gauss = cv2.GaussianBlur(gray, (3, 3), 1)
            gaus = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
            image = cv2.cvtColor(gaus, cv2.COLOR_GRAY2RGB)/255.0
        
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        self.net.forward()

        probs = [self.net.blobs[layername].data[0].flatten() for layername in layernames]
        index_probs = [prob.argsort()[-1] for prob in probs]

        return probs, index_probs
