#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
import math
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))
sys.path.append(os.path.join(tool_root, 'utils'))
os.environ['GLOG_minloglevel'] = '2'
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
import cv2
import shutil
from lstm import CaffeLstmClassification
from lstm import get_ctc_decoder
from file2list import readTxt


Nummap = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', ' ']

def parse_args():
    '''银行卡卡号/日期识别模型参数设置'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/ocr/card/model/owner/V0.0.0/model/LowercaseNumberRecogLSTM.prototxt')
    parser.add_argument('--image_resize', default=[40, 400], type=int)
    parser.add_argument('--mean_value', default=[174, 173, 171])
    parser.add_argument('--time_step', default=100)
    parser.add_argument('--resize_type', default='undeform_center_imgmean')
    parser.add_argument('--model_weights',
                        default='/work/ocr/card/model/owner/V0.0.0/model/owner_iter_243000.caffemodel')
    return parser.parse_args()

def bank_card_recognise_lstm(img_dir, labelindex=0, save_error_dir=''):
    args = parse_args()
    lstm_classification = CaffeLstmClassification(args.gpu_id, args.model_def, args.model_weights, args.time_step,
                                                  args.image_resize, args.mean_value, resize_type=args.resize_type)
    IsFile, IsSave = False, False
    if os.path.isfile(img_dir):
        img_list = readTxt(img_dir)
        IsFile = True
    elif os.path.isdir(img_dir):
        img_list = os.listdir(img_dir)
    else:
        logging.info("Error: 不支持的路径输入 -> %s" %(img_dir))

    logging.info("Deal with the file: %s[size: %s]" %(os.path.basename(img_dir), len(img_list)))
    logging.info("Model: %s" %(os.path.basename(args.model_weights)))

    if len(save_error_dir) != 0:
        IsSave = True
        if not os.path.exists(save_error_dir):
            os.mkdir(save_error_dir)
        logging.info("错误图片将会保存，保存地址为: %s" %(save_error_dir))
    
    right_cnt, cnt = 0, 0
    
    for item in img_list:
        cnt += 1
        image_file = os.path.join(img_dir, item)
        if IsFile:
            image_file = item
        save_pre = os.path.basename(item).split('.jpg')[0]
        label = save_pre.split('_')[labelindex]
        _, prob_index = lstm_classification.classify(image_file, 'premuted_fc')
        pro_label = get_ctc_decoder(prob_index, Nummap)

        if label == pro_label:
            right_cnt += 1
        else:
            if IsSave:
                save_path = os.path.join(save_error_dir,  pro_label + '_' + item)
                shutil.copy(image_file, save_path)
            else:
                logging.info("[%s]%s,%s,%s" %(cnt, image_file, label, pro_label))
    logging.info("all cont: %s, right num: %s\n acc: %s" %(cnt, right_cnt, right_cnt*1.0/cnt))

def main():
    img_dir = '/work/ocr/card/vehicle_license/data/test/cut/1PlateNo/ok'
    bank_card_recognise_lstm(img_dir, labelindex=0, save_error_dir='')

if __name__ == '__main__':
    main()