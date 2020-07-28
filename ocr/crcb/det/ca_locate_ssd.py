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

##Chn前向预测的ssd检测定位
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/data1/ocr/ocr_project/SignatureRec/model/CapitalMoneyRecog/labelmap.prototxt')
    parser.add_argument('--mean_value', default=[185, 185, 185])
    parser.add_argument('--input_scale', default=1)
    parser.add_argument('--model_def',
                        default='/data1/ocr/net/SSD/CA/deploy.prototxt')
    parser.add_argument('--image_resize', default=[160, 480], type=int)
    parser.add_argument('--model_weights',
                        default='/data1/ocr/net/SSD/CA/weights/shred_chn_print_iter_14000.caffemodel')
    return parser.parse_args()

##beifen
def parse_args_bak():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/data1/ocr/ocr_project/SignatureRec/model/CapitalMoneyRecog/labelmap.prototxt')
    parser.add_argument('--mean_value', default=[185, 185, 185])
    parser.add_argument('--input_scale', default=1)
    parser.add_argument('--model_def',
                        default='/data1/ocr/net/SSD/CA/deploy.prototxt')
    parser.add_argument('--image_resize', default=[160, 480], type=int)
    parser.add_argument('--model_weights',
                        default='/data1/ocr/net/SSD/CA/weights/shred_chn_print_iter_14000.caffemodel')
    return parser.parse_args()


def main(test_root_dir):
    args = parse_args_bak()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file, args.mean_value, args.input_scale)
    version = 'V0.0.0'
    save_dir = os.path.join(test_root_dir, '..', version)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for item in os.listdir(test_root_dir):
        image_file = os.path.join(test_root_dir, item)
        save_path = os.path.join(save_dir, item)
        flage = get_bnbox(image_file, detection, save_path, False)
        # print(flage)


if __name__ == '__main__':
    test_root_dir = '/data1/ocr/data/Chn/20180802/20180802_print'
    main(test_root_dir)
    print(test_root_dir)

    # test_root_dir = '/work/hena/ocr/data/CLPR/det/test/shouchi/baoding/yolo_car'
    # main(test_root_dir)
    # print(test_root_dir)

    # test_root_dir = '/work/hena/ocr/data/CLPR/det/test/shouchi/suzhou/test'
    # main(test_root_dir)
    # print(test_root_dir)

    
    # test_root_dir = '/work/hena/ocr/data/CLPR/det/tmo'
    # main(test_root_dir)
    # print(test_root_dir)

    # test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/body/ori'
    # main(test_root_dir)
    # print(test_root_dir)

    # test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/lower/1204cars'
    # main(test_root_dir)
    # print(test_root_dir)
    
    # test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/normal/ori'
    # main(test_root_dir)
    # print(test_root_dir)

    # test_root_dir = '/work/hena/ocr/data/CLPR/det/test/specialpics/otherwords/ori'
    # main(test_root_dir)
    # print(test_root_dir)