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
from ssd_detect import get_bnbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/hena/ocr/model/caffe/CLPR/det/V0.2.0/model/labelmap_voc.prototxt')
    parser.add_argument('--mean_value', default=[106.5, 101.7, 95.4])
    parser.add_argument('--input_scale', default=0.0039215)
    parser.add_argument('--model_def',
                        default='//work/hena/ocr/model/caffe/CLPR/det/V0.3.0/model/clpr_det_v031_deploy.prototxt')
    parser.add_argument('--image_resize', default=[192, 192], type=int)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/CLPR/det/V0.3.0/weights_bak/clpr_det_v031_iter_41100.caffemodel')
    return parser.parse_args()

def parse_args_bak():
    '''parse args
    input 的图像尺度到[-1, 1]'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/hena/ocr/model/caffe/CLPR/det/V0.2.0/model/labelmap_voc.prototxt')
    parser.add_argument('--mean_value', default=[127.5, 127.5, 127.5])
    parser.add_argument('--input_scale', default=0.0078125)
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/CLPR/det/V0.3.0/model/clpr_det_v030_deploy.prototxt')
    parser.add_argument('--image_resize', default=[192, 192], type=int)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/CLPR/det/V0.3.0/model/clpr_det_v030_iter_55000.caffemodel')
    return parser.parse_args()

def main(test_root_dir):
    args = parse_args()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file, args.mean_value, args.input_scale)
    version = 'V0.3.1'
    save_dir = os.path.join(test_root_dir, '..', version)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for item in os.listdir(test_root_dir):
        image_file = os.path.join(test_root_dir, item)
        save_path = os.path.join(save_dir, item)
        flage = get_bnbox(image_file, detection, save_path, False)


if __name__ == '__main__':
    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/plate_cls/double_yellow/images'
    main(test_root_dir)
    print(test_root_dir)

    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/shouchi/baoding/yolo_car'
    main(test_root_dir)
    print(test_root_dir)

    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/shouchi/suzhou/test'
    main(test_root_dir)
    print(test_root_dir)

    
    test_root_dir = '/work/hena/ocr/data/CLPR/det/tmo'
    main(test_root_dir)
    print(test_root_dir)

    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/body/ori'
    main(test_root_dir)
    print(test_root_dir)

    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/lower/1204cars'
    main(test_root_dir)
    print(test_root_dir)
    
    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/normal/ori'
    main(test_root_dir)
    print(test_root_dir)

    test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/otherwords/ori'
    main(test_root_dir)
    print(test_root_dir)