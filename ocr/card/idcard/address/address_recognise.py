#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
import argparse
import cv2
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from chars_recognise import get_classification
from chars_recognise import chars_recognise
from chars_segment import get_chars_bnboxs
from resize_img import undeform_center_resize
from resize_img import map4undeform_center_resize


def chars_cls_args():
    '''
    chars_cls model argument
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='', type=str, help='需要分类的图片根目录')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--mean_value', default=[127.5, 127.5, 127.5])
    parser.add_argument('--scale', default=0.0078125)
    parser.add_argument('--resize_type', default='adaptiveBinary', help='adaptiveBinary')
    parser.add_argument('--hard_list', default='/work/help/szp/chars/model/nan.txt')
    parser.add_argument('--easy_list', default='/work/help/szp/chars/model/yi.txt')
    parser.add_argument('--label_list', default='/work/help/szp/chars/model/label.txt')
    parser.add_argument('--model_def',
                        default='/work/help/szp/chars/model/deploy_train_zifu.prototxt')
    parser.add_argument('--image_resize', default=[24, 24], type=int)
    parser.add_argument('--model_weights',
                        default='/work/help/szp/chars/model/model_iter_36000.caffemodel')
    return parser.parse_args()

def chars_seg_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--img_dir', type=str, default='/work/ocr/card/driving_license/data/seg/test', help='需要分割的图片根目录')
    parser.add_argument('--labelmap_file',
                        default='/work/ocr/card/model/chars_segment/V0.0.1/model/chars_seg_v001_labelmap_voc.prototxt')
    parser.add_argument('--mean_value', default=[145, 140, 144])
    parser.add_argument('--input_scale', default=1)
    parser.add_argument('--model_def',
                        default='/work/ocr/card/model/chars_segment/V0.0.1/model/chars_seg_v001_deploy.prototxt')
    parser.add_argument('--image_resize', default=[300, 300], type=int, help='[resized_h, resized_w]')
    parser.add_argument('--resize_type', default='undeform_center', type=str)
    parser.add_argument('--model_weights',
                        default='/work/ocr/card/model/chars_segment/V0.0.1/model/chars_seg_v001_iter_140000.caffemodel')
    return parser.parse_args()

def address_recognise(img_dir, _IS_SAVE_SEG_=False, _IS_DEBUG_CLS_=False, showkey=False, version='char_seg_v000'):
    args = chars_cls_args()
    seg_args = chars_seg_args()
    classification = get_classification(args)
    # chars_bnbox_list = [[xmin, xmax, ymin, ymax, label], ...]
    chars_bnbox_list, img_path_list = get_chars_bnboxs(seg_args, img_dir, _IS_SAVE_SEG_, showkey, version)

    results = []
    for i, item in enumerate(chars_bnbox_list):
        result = []
        img_path = os.path.join(img_dir, img_path_list[i])
        img = cv2.imread(img_path)
        for char in item:
            xmin, xmax, ymin, ymax = char[: 4]
            chars_img = img[ymin: ymax, xmin: xmax]
            prob_char, prob, probs, index_probs = chars_recognise(chars_img, classification, args, _IS_DEBUG_CLS_)
            result.append(prob_char)
        results.append(result)
    return results, img_path_list


if __name__ == '__main__':
    results, img_path_list= address_recognise('/work/ocr/card/vehicle_license/data/test/cut/3Owner/test/tmp1',  _IS_DEBUG_CLS_=False, showkey=False)
    for i, result in enumerate(results):
        print('img_name: %s\t rec result: %s' %(img_path_list[i], ''.join(j for j in [k for k in result if k != 'none'])))



    