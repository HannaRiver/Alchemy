#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
分类器只适用于TinyMind人民币分类
'''
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))
os.environ['GLOG_minloglevel'] = '2'
import argparse
from classify_class import CaffeClassification
import numpy as np
import shutil
import cv2



def get_right_label(image_file, classification):
    _, index_prob = classification.classify(image_file)

    image_name = os.path.basename(image_file)
    if index_prob == 1:
        shutil.copy(image_file, os.path.join(check_dir + 'handwriting', image_name))
    else:
        shutil.copy(image_file, os.path.join(check_dir + 'print', image_name))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/competitions/TinyMind/net/AlexNet/model/deploy.prototxt')
    parser.add_argument('--image_resize', default=[150, 300], type=int)
    parser.add_argument('--mean_value', default=[120, 127, 130])
    parser.add_argument('--input_scale', default=0.00392156862745098)
    parser.add_argument('--resize_type', default='')
    parser.add_argument('--model_weights',
                        default='/work/competitions/TinyMind/net/AlexNet/weights/RMBType_iter_5000.caffemodel')
    return parser.parse_args()

def main(args, image_dir):
    classification = CaffeClassification(args.gpu_id,
                                         args.model_def, args.model_weights,
                                         args.image_resize, args.mean_value,
                                         args.input_scale, (2, 0, 1), args.resize_type)
    cnt_right, cnt_all = 0, 0
    maplist = ['0.1', '0.2', '0.5', '1', '2', '5', '10', '50', '100']
    with open(image_dir, 'r') as f:
        for line in f.readlines():
            image_path, gt_label = line.strip().split(' ')
            # if not os.path.exists(image_path):
            #     continue
            cnt_all += 1
            _, index_prob = classification.classify(image_path, 'prob')
            if str(index_prob) == gt_label:
                cnt_right += 1
            else:
                print(image_path, gt_label, index_prob)
                # print(maplist[index_prob])
                # show_img = cv2.imread(image_path)
                # cv2.namedWindow("char segment result", cv2.WINDOW_NORMAL)
                # cv2.imshow("char segment result", show_img)
                # cv2.waitKey(0)
    print(cnt_right*1.0/cnt_all)

def main_pre(args, image_dir, save_path):
    classification = CaffeClassification(args.gpu_id,
                                         args.model_def, args.model_weights,
                                         args.image_resize, args.mean_value,
                                         args.input_scale, (2, 0, 1), args.resize_type)
    fw = open(save_path, 'w')
    fw.write('name, label\n')
    maplist = ['0.1', '0.2', '0.5', '1', '2', '5', '10', '50', '100']
    for img_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, img_name)
        _, index_prob = classification.classify(image_path, 'prob')
        pre_lable = maplist[index_prob]
        save_item = img_name + ',' + pre_lable + '\n'
    fw.close()

if __name__ == '__main__':
    args = parse_args()
    # main(args, data_root)
    image_dir = '/work/competitions/TinyMind/public_test_data'
    save_path = '/work/competitions/TinyMind/result'
    main_pre(args, image_dir, save_path)