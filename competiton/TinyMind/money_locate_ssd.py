#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
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


def parse_args_SSD():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/competitions/TinyMind/net/SSD/model/labelmap.prototxt')
    parser.add_argument('--mean_value', default=[120, 127, 130])
    parser.add_argument('--input_scale', default=1)
    parser.add_argument('--model_def',
                        default='/work/competitions/TinyMind/net/SSD/model/deploy.prototxt')
    parser.add_argument('--image_resize', default=[300, 300], type=int)
    parser.add_argument('--model_weights',
                        default='/work/competitions/TinyMind/net/SSD/model/weights/money_det_001_iter_12000.caffemodel')
    return parser.parse_args()

def parse_args_CLPRSSD():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/competitions/TinyMind/net/CLPRSSD/model/labelmap.prototxt')
    parser.add_argument('--mean_value', default=[127.5, 127.5, 127.5])
    parser.add_argument('--input_scale', default=0.0078125)
    parser.add_argument('--model_def',
                        default='/work/competitions/TinyMind/net/CLPRSSD/model/FPNdeploy.prototxt')
    parser.add_argument('--image_resize', default=[192, 192], type=int)
    parser.add_argument('--model_weights',
                        default='/work/competitions/TinyMind/net/CLPRSSD/weights/CLPRSSD_iter_8000.caffemodel')
    return parser.parse_args()

def main(test_root_dir):
    args = parse_args_CLPRSSD()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file, args.mean_value, args.input_scale)
    key = os.path.basename(test_root_dir)
    version = 'CLPRSSD_' + key + '_8k'
    save_dir = os.path.join(test_root_dir, '..', version)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for item in os.listdir(test_root_dir):
        image_file = os.path.join(test_root_dir, item)
        save_path = os.path.join(save_dir, item)
        flage = get_bnbox(image_file, detection, save_path, False)

def test_single_image(image_file, save_path):
    args = parse_args_bak()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file, args.mean_value, args.input_scale)
    flage = get_bnbox(image_file, detection, save_path, True)

if __name__ == '__main__':
    # for i in ['0.1', '0.2', '0.5', '1.0', '2.0', '5.0', '10.0', '50.0', '100.0']:
    #     test_root_dir = '/work/competitions/TinyMind/result/' + i
    #     save_path = ''
    #     main(test_root_dir)
    test_root_dir = '/work/competitions/TinyMind/private_test_data'
    main(test_root_dir)
