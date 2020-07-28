#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))
os.environ['GLOG_minloglevel'] = '2'

import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

from classify_class import CaffeClassification
import cv2

_IS_DEBUG_ = 1


def parse_args():
    '''parse args
    input 的图像尺度到[-1, 1]'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--mean_value', default=[127.5, 127.5, 127.5])
    parser.add_argument('--scale', default=0.0078125)
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/CLPR/cls/car_head_tail.prototxt')
    parser.add_argument('--image_resize', default=[80, 80], type=int)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/CLPR/cls/car_head_tail.caffemodel')
    return parser.parse_args()

def car_head_cls():
    args = parse_args()
    classification = CaffeClassification(args.gpu_id, args.model_def, args.model_weights, args.image_resize, args.mean_value, args.scale)

    img_root = '/work/hena/ocr/data/CLPR/det/test/images'
    for item in os.listdir(img_root):
        image_file = os.path.join(img_root, item)
        prob, index_prob = classification.classify(image_file, 'loss3/prob')
        
        print index_prob, prob
        if _IS_DEBUG_:
            cv2.namedWindow("cls_result", 0)
            cv2.imshow("cls_result", cv2.imread(image_file))
            cv2.waitKey(0)

if __name__ == "__main__":
    car_head_cls()
    
