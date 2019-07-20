#-*- coding=utf-8 -*-
'''
Detection with normal SSD
In this example, we will load a SSD model and use it to detect objects.
Support TextBoxes++ Version(detect_quad)
Author: He Na
Time: 2018.12.29
'''

from __future__ import division
import os
import sys
import cv2
import numpy as np
import shutil
from alchemy_config import cfg
utils_dir = cfg.UTILS_DIR
caffe_root = cfg.CAFFE_ROOT.SSD
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
sys.path.append(utils_dir)
os.environ['GLOG_minloglevel'] = '2'
from resize_img import undeform_center_resize



def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file, mean_value=[0, 0, 0], input_scale=1, resize_type=''):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize_h = image_resize[0]
        self.image_resize_w = image_resize[1]
        self.resize_type = resize_type
        self.mean_value = mean_value
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array(mean_value)) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_input_scale('data', input_scale)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.2, topn=1):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        image = caffe.io.load_image(image_file)

        if self.resize_type == 'undeform_center':
            image = undeform_center_resize(image*255, self.image_resize_h, self.image_resize_w, self.mean_value)/255.0

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        # top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
        my_top_indice = [(i, conf) for i, conf in enumerate(det_conf) if conf>=conf_thresh]
        my_top_indice.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in my_top_indice]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        ################ hena #######################
        label_cls = list(set(top_label_indices))
        final_list = []
        for item in label_cls:
            lable_index = [i for i, cls_info in enumerate(top_label_indices) if cls_info==item]
            lable_index = lable_index[: topn]
            final_list += lable_index
        #############################################

        result = []
        # for i in xrange(min(topn, top_conf.shape[0])):
        for i in final_list:
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result
    
    def detect_quad(self, image_file, conf_thresh=0.2, topn=1):
        '''
        TextBoxes++
        QUAD point
        '''
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        image = caffe.io.load_image(image_file)

        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']

        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        det_x1 = detections[0,0,:,7]
        det_y1 = detections[0,0,:,8]
        det_x2 = detections[0,0,:,9]
        det_y2 = detections[0,0,:,10]
        det_x3 = detections[0,0,:,11]
        det_y3 = detections[0,0,:,12]
        det_x4 = detections[0,0,:,13]
        det_y4 = detections[0,0,:,14]

        my_top_indice = [(i, conf) for i, conf in enumerate(det_conf) if conf>=conf_thresh and 1>det_x1[i]>0 and 1>det_y1[i]>0
         and 0<det_x2[i]<1 and 0<det_y2[i]<1 and 0<det_x3[i]<1 and 0<det_y3[i]<1 and 0<det_x4[i]<1 and 0<det_y4[i]<1]
        my_top_indice.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in my_top_indice]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        top_x1 = det_x1[top_indices]
        top_y1 = det_y1[top_indices]
        top_x2 = det_x2[top_indices]
        top_y2 = det_y2[top_indices]
        top_x3 = det_x3[top_indices]
        top_y3 = det_y3[top_indices]
        top_x4 = det_x4[top_indices]
        top_y4 = det_y4[top_indices]

        ################ hena #######################
        label_cls = list(set(top_label_indices))
        final_list = []
        for item in label_cls:
            lable_index = [i for i, cls_info in enumerate(top_label_indices) if cls_info==item]
            lable_index = lable_index[: topn]
            final_list += lable_index
        #############################################
        result = []
        # for i in xrange(min(topn, top_conf.shape[0])):
        for i in final_list:
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            result.append([xmin, ymin, xmax, ymax, label, score])
        return result
    
def get_bnbox(image_file, detection, save_path='', showkey=False, thr=0.0, topn=1, is_rectimg=False):
    results = detection.detect(image_file, thr, topn)
    if len(results) == 0:
        if len(save_path) != 0:
            pass
            # shutil.copy(image_file, save_path)
        return False
    
    img = cv2.imread(image_file)
    img2 = img.copy()
    save_img = ''
    for result in results:
        label = result[4]
        
        height, width = img.shape[: 2]
        xmin, xmax = [int(round(result[i] * width)) for i in [0, 2]]
        ymin, ymax = [int(round(result[i] * height)) for i in [1, 3]]

        if xmin < 0 or ymin < 0 or ymax <= ymin or xmax <= xmin:
            continue
        roi_img = img2[ymin: ymax, xmin: xmax]

        color_dic = {'11': (255, 0, 0), '14': (0, 255, 0), '19': (0, 0, 255), '20': (255, 255, 0)}
        color = (0, 255, 255) if str(label) not in color_dic else color_dic[str(label)]
        save_img = roi_img
        if label == 1 or True:
            # save_img = roi_img.copy()
            score = str(result[5])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            # label_name = result[-1]
            label_name = 'Plate'
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, (label_name+" score: " + score), (xmin-5, ymin+5), font, 2, (0, 255, 0), 2)
        if len(save_path) != 0 and save_img != '' and is_rectimg:
            img_locate = '_' + str(xmin) + '_' + str(xmax) + '_' + str(ymin) + '_' + str(ymax) + '_' + str(label)
            save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path)[: -4] + img_locate + os.path.basename(save_path)[-4: ])
            cv2.imwrite(save_path, save_img)
        if showkey:
            cv2.namedWindow("ssd result", cv2.WINDOW_NORMAL)
            cv2.imshow("ssd result", img)
            cv2.waitKey(0)
    if not is_rectimg and len(save_path) != 0:
        cv2.imwrite(save_path, img)

    return True

def get_bnbox_quad(image_file, detection, save_path='', showkey=False, thr=0.0, topn=1, is_rectimg=False):
    results = detection.detect_quad(image_file, thr, topn)
    if len(results) == 0:
        if not showkey:
            shutil.copy(image_file, save_path)
        return False
    
    img = cv2.imread(image_file)
    for result in results:
        label = result[4]
        
        height, width = img.shape[: 2]
        xmin, xmax = [int(round(result[i] * width)) for i in [0, 2]]
        ymin, ymax = [int(round(result[i] * height)) for i in [1, 3]]

        if xmin < 0 or ymin < 0 or ymax <= ymin or xmax <= xmin:
            continue

        color = (0, 0, 255) if label == 1 else (255, 0, 0)
        if label == 1 or True:
            score = str(result[5])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, ("score: "+ score), (10, 50), font, 1, (0, 0, 255), 1)

    if showkey:
        cv2.namedWindow("ssd result", cv2.WINDOW_NORMAL)
        cv2.imshow("ssd result", img)
        cv2.waitKey(0)
    if not is_rectimg and len(save_path) != 0:
        cv2.imwrite(save_path, img)
    
    return True