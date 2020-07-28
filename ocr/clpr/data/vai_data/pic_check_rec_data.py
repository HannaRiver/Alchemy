#/usr/bin/python
#-*- encoding=utf-8 -*-

import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))
sys.path.append(os.path.join(tool_root, 'utils'))
os.environ['GLOG_minloglevel'] = '2'

import argparse
import shutil
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

from lstm import CaffeLstmClassification
from lstm import get_ctc_decoder
from file2list import readTxt
from char_trans import del_chn


CLPRmap = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', '京', '沪', '津', '渝', '黑', '吉',
           '辽', '蒙', '冀', '新', '甘', '青', '陕', '宁', '豫', '鲁',
           '晋', '皖', '鄂', '湘', '苏', '川', '贵', '云', '桂', '藏',
           '浙', '赣', '粤', '闽', '琼', '挂', '学', '警', ' ']

def parse_args():
    '''车牌识别模型参数设置'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='/work/hena/ocr/data/CLPR/VAI/batch4_20190123/plate_error')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/CLPR/rec/V1.0.1/model/V1.0.1.2/deploy.prototxt')
    parser.add_argument('--image_resize', default=[112, 240], type=int)
    parser.add_argument('--mean_value', default=[0, 0, 0])
    parser.add_argument('--time_step', default=60)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/CLPR/rec/V1.0.1/model/V1.0.1.4/clpr_rec_v1014_132000.caffemodel')
    return parser.parse_args()


def plate_recognize_lstm(img_dir, save_right_dir, save_error_dir):
    args = parse_args()
    lstm_classification = CaffeLstmClassification(args.gpu_id, args.model_def, args.model_weights, args.time_step,
                                                  args.image_resize, args.mean_value)
    
    img_list = os.listdir(img_dir)
    logging.info("Deal with the file: %s[size: %s]" %(img_dir, len(img_list)))
    right_cnt, cnt = 1, 1
    
    for item in img_list:
        cnt += 1
        image_file = os.path.join(img_dir, item)
        if '.jpg' not in os.path.basename(item):
            logging.info("Warring:: image must with .jpg -> %s" %(item))
            continue
        save_pre = os.path.basename(item).split('.jpg')[0]
        label = save_pre.split('_')[1]
        _, prob_index = lstm_classification.classify(image_file, 'premuted_fc')
        pro_label = get_ctc_decoder(prob_index, CLPRmap)

        if del_chn(label) == del_chn(pro_label):
            right_cnt += 1
            shutil.copy(image_file, os.path.join(save_right_dir, pro_label + '_' + item))
        else:
            think_label = pro_label[: len(pro_label) - len(label)] + label
            save_path = os.path.join(save_error_dir, think_label + '_' + item)
            shutil.copy(image_file, save_path)
    logging.info("%s-acc: %s \n" %(img_dir, right_cnt*1.0/cnt))

def vai_main():
    img_root = parse_args().img_root
    img_dir_list = os.listdir(img_root)
    logging.info("Deal with the vai data ==>> %s" %(img_root))
    for item in img_dir_list:
        logging.info("========== %s ==========" %(item))
        img_dir = os.path.join(img_root, item)
        save_right_dir = os.path.join(img_root, item + '_right')
        save_error_dir = os.path.join(img_root, item + '_error')

        if not os.path.exists(save_error_dir):
            os.mkdir(save_error_dir)
            os.mkdir(save_right_dir)
        
        plate_recognize_lstm(img_dir, save_right_dir, save_error_dir)

if __name__ == '__main__':
    vai_main()
