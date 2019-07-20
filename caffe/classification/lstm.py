#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
Classification with LSTM caffe
Author: He Na
Time: 2018.12.29
'''
import os
import sys
from alchemy_config import cfg
utils_dir = cfg.UTILS_DIR
caffe_root = cfg.CAFFE_ROOT.LSTM
sys.path.insert(0, utils_dir)
import cv2
import numpy as np
import shutil
from file2list import readTxt
from resize_img import undeform_center_resize, resize_with_center_pad, undeform_resize
os.environ['GLOG_minloglevel'] = '2'
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe


class CaffeLstmClassification:
    def __init__(self, gpu_id, model_def, model_weights, time_step, image_resize, mean_value=[0, 0, 0], input_scale=1, resize_type=''):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize_h = image_resize[0]
        self.image_resize_w = image_resize[1]
        self.mean_value = mean_value
        self.resize_type = resize_type
        self.net = caffe.Net(model_def,
                             model_weights,
                             caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array(mean_value))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_input_scale('data', input_scale)
        self.transformer.set_channel_swap('data', (2, 1, 0))

        self.time_step = time_step
    
    def classify(self, image_file, layers='premuted_fc'):
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        if self.resize_type == '':
            image = caffe.io.load_image(image_file) if type(image_file) == type('hello') else image_file / 255.0
        else:
            image = cv2.imread(image_file) if type(image_file) == type('Hi') else image_file      
            if self.resize_type == 'undeform_center_imgmean':
                mean_value = [np.mean(image[:,:,i]) for i in [0, 1, 2]]
                image = resize_with_center_pad(image, self.image_resize_h, self.image_resize_w, mean_value)/255.0
            elif self.resize_type == 'undeform_center':
                image = undeform_center_resize(image, self.image_resize_h, self.image_resize_w, self.mean_value)/255.0
            elif self.resize_type == 'undeform_resize':
                image = undeform_resize(image, self.image_resize_h, self.image_resize_w, self.mean_value)/255.0     
            elif self.resize_type == 'undeform_resize_white':
                image = undeform_resize(image, self.image_resize_h, self.image_resize_w, [255, 255, 255])/255.0
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        self.net.forward()

        prob = self.net.blobs[layers].data[0].flatten()
        prob = vector2array(prob, self.time_step)
        # print len(vector2array(prob))
        index_prob = [prob[i].argsort()[-1] for i in range(len(prob))]
        return prob, index_prob

def vector2array(avector, time_step=80):
    assert(len(avector)%time_step == 0), 'vector lenth error!'
    cls_num = len(avector)/time_step
    return [avector[cls_num*i: cls_num*(i+1)] for i in range(time_step)]

def get_ctc_decoder(alist, maplist, black_id=-1):
    if black_id == -1:
        black_id = maplist.index(' ')
    rlist = [alist[0]]
    for i in range(len(alist)-1):
        if alist[i+1] == alist[i]:
            continue
        rlist.append(alist[i+1])
    rrlist = []
    for i in rlist:
        if i == black_id:
            continue
        rrlist.append(i)
    return str('').join(maplist[i] for i in rrlist)