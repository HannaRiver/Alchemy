#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'detection'))
os.environ['GLOG_minloglevel'] = '2'

import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

import cv2
import shutil
from ssd_detect import CaffeDetection
from ssd_detect import get_bnbox_quad

def parse_args():
    '''parse args
    input 的图像尺度到[-1, 1]'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/hena/ocr/model/caffe/CLPR/det/v0.0.2/labelmap_voc.prototxt')
    parser.add_argument('--mean_value', default=[104, 117, 123])
    parser.add_argument('--scale', default=1)
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/CLPR/det/texboxes/deploy.prototxt')
    parser.add_argument('--image_resize', default=[384, 384], type=int)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/CLPR/det/texboxes/VGG_text_text_polygon_precise_fix_order_384x384_iter_120000.caffemodel')
    return parser.parse_args()

def main():
    args = parse_args()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file, args.mean_value, args.scale)
    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/lower/1204cars'
    version = 'V0.0.1'
    save_dir = os.path.join(test_root_dir, '..', version)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for item in os.listdir(test_root_dir):
        image_file = os.path.join(test_root_dir, item)
        save_path = os.path.join(save_dir, item)
        flage = get_bnbox_quad(image_file, detection, save_path, True)


if __name__ == '__main__':
    main()

    
