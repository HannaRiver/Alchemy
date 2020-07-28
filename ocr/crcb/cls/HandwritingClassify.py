#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
分类器只适用于常熟碎片手写非手写分类
'''
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))
os.environ['GLOG_minloglevel'] = '2'
import argparse
from classify_class import CaffeClassification
import numpy as np
import shutil



def get_right_label(image_file, classification, check_dir):
    _, index_prob = classification.classify(image_file)

    image_name = os.path.basename(image_file)
    if index_prob == 1:
        shutil.copy(image_file, os.path.join(check_dir + 'handwriting', image_name))
    else:
        shutil.copy(image_file, os.path.join(check_dir + 'print', image_name))

def parse_args():
    '''crcb handwriting classication args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', default='付款人户名')
    parser.add_argument('--date', default='20180802')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/HandWriting/isHW/model/crcb_shred_deploy.prototxt')
    parser.add_argument('--image_resize', default=[192, 512], type=int)
    parser.add_argument('--mean_value', default=[205, 203, 202])
    parser.add_argument('--input_scale', default=1)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/HandWriting/isHW/weights/crcb_shred_cls_iter_450000.caffemodel')
    return parser.parse_args()

def main(args, image_dir, check_dir):
    classification = CaffeClassification(args.gpu_id,
                                         args.model_def, args.model_weights,
                                         args.image_resize, args.mean_value,
                                         args.input_scale)
    
    for i in ['_handwriting/', '_print/']:
        if not os.path.exists(os.path.join(check_dir, args.date + i)):
            os.mkdir(os.path.join(check_dir, args.date + i))

    check_dir = os.path.join(check_dir, args.date + '_')

    for image_file in [os.path.join(image_dir, i) for i in os.listdir(image_dir)]:
        get_right_label(image_file, classification, check_dir)

if __name__ == '__main__':
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item'
    args = parse_args()
    check_dir = os.path.join(data_root, args.item)
    image_dir = os.path.join(data_root, args.item, args.date)
    
    print(image_dir)

    main(args, image_dir, check_dir)