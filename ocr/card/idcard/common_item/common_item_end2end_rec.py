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
import pickle

chejian_old = open('/work/ocr_base/data/ls_general_gwx/model/alphabet_chejian_old.pkl', 'rb')
ls_general = open('/home/shenzhou/caffe/caffe-lstm/models/ls_general/alphabet_ls_general.pkl', 'rb')
ICDAR = open('/work/caffe_lstm_train_project/lstm_dataset/ICDAR/alphabet.pkl','rb')


# Labelmap = pickle.load(chejian_old)
# Labelmap = pickle.load(ls_general)
Labelmap = pickle.load(ICDAR)



def parse_args():
    '''姓名端对端识别模型参数设置'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/home/shenzhou/caffe/caffe-lstm/models/ICDAR/deploy.prototxt')
    # parser.add_argument('--model_def',
                        # default='/work/codes/0219/chejian_v5/model/normal/ls_general_tmp.prototxt')                        
    parser.add_argument('--image_resize', default=[32, 320], type=int)
    parser.add_argument('--mean_value', default=[0, 0, 0])
    parser.add_argument('--time_step', default=80)
    parser.add_argument('--resize_type', default='padding_white')
    parser.add_argument('--model_weights',
                        default='/work/caffe_lstm_train_project/caffemodel/ICDAR/0314_models/ls_ICDAR_snapshot__iter_73207.caffemodel')
    # parser.add_argument('--model_weights',
                        # default='/work/caffe_lstm_train_project/caffemodel/ls_general/0312_models/ls_general_snapshot__iter_72192.caffemodel')# V0.03_0312
    # parser.add_argument('--model_weights',
                        # default='/work/caffe_lstm_train_project/caffemodel/ls_general/0304_models/ls_general_snapshot__iter_161212.caffemodel')# V0.02
    # parser.add_argument('--model_weights',
                        # default='/work/codes/0219/chejian_v5/model/normal/ls_general.caffemodel')
    parser.add_argument('--label_dict', default='/work/ocr/card/model/owner/V0.0.0/model/ditc.pkl')
    return parser.parse_args()

def common_item_recognise_lstm(img_dir, labelindex=0, save_error_dir=''):
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
        # label = save_pre
        # print(label)
        # label = ''
        _, prob_index = lstm_classification.classify(image_file, 'premuted_fc')
        pro_label = get_ctc_decoder(prob_index, Labelmap, 0)
        # pro_label = get_ctc_decoder(prob_index, Labelmap, 7117)
        # pro_label = pro_label.replace("-","")
        pro_label = pro_label.replace(' ','')
        
        # label = '住址'
        print(label)
        print(pro_label)
        if label == pro_label:
            right_cnt += 1
        else:
            if IsSave:
                save_path = os.path.join(save_error_dir,  pro_label + label + '_' + item)
                # shutil.copy(image_file, save_path)
            else:
                logging.info("[%s]%s,%s,%s" %(cnt, image_file, label, pro_label))
    logging.info("all cont: %s, right num: %s\n acc: %s" %(cnt, right_cnt, right_cnt*1.0/cnt))

def main():
    # img_root = '/work/ocr_base/data/ls_general_gwx/xsz_data_plus/drving_card_dataset_20190228/unfixed_total/5UseCharacter_changed'
    # for item in os.listdir(img_root):
    #     img_dir = os.path.join(img_root, item)
    #     common_item_recognise_lstm(img_dir, labelindex=-1, save_error_dir='')
    img_dir = '/work/ocr_base/data/test/4_test_on_jsz_fake/template1/出生日期'
    common_item_recognise_lstm(img_dir, labelindex=1, save_error_dir='/work/ocr_base/data/test/erro_tmp')

if __name__ == '__main__':
    main()