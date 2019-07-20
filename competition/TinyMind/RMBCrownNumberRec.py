#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
这个代码主要实现的功能为: TinyMind人民币冠号识别
'''

import os
import sys
import cv2
import math
import shutil
from functools import reduce
from decimal import Decimal
import glob
from alchemy_config import cfg
tool_root = cfg.ROOT_DIR
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))
sys.path.append(os.path.join(tool_root, 'utils'))
os.environ['GLOG_minloglevel'] = '2'

import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

from lstm import CaffeLstmClassification


def SoftMax(net_ans):
    tmp_net = [math.exp(i) for i in net_ans]
    sum_exp = sum(tmp_net)
    return [i/sum_exp for i in tmp_net]

RMBmap = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Q', 'W', 'E', 'R', 'A', 'H', 'S', 'X', 'J', 'Y', 'T', 'O', 'B', 'U', 'P', 'Z', 'C', 'L', 'F', 'G', 'D', 'M', 'I', 'K', 'N']
def readTxt(txtpath):
    filelist = {}
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            label, item = line.strip().split('_')[: 2]
            filelist[item] = label
    return filelist
ok_dict = readTxt('/work/competitions/TinyMind/data/finals/ok.txt')


def get_ctc_decoder_refuse(prob_index, prob, maplist, black_id=0):
    black_id = maplist.index(' ')
    rlist = [prob_index[0]]
    rprob = [prob[0][prob_index[0]]]
    for i in range(len(prob_index)-1):
        if prob_index[i+1] == prob_index[i]:
            tmp_score = prob[i+1][prob_index[i+1]]
            if tmp_score > rprob[-1]:
                rprob[-1] = tmp_score
            continue
        rlist.append(prob_index[i+1])
        rprob.append(prob[i+1][prob_index[i+1]])
    rrlist = []
    rrprob = []
    for i, item in enumerate(rlist):
        if item == black_id:
            continue
        rrlist.append(item)
        rrprob.append(rprob[i])
    pre_result = str('').join(maplist[i] for i in rrlist)
    return pre_result, rrprob, rlist, rprob

def parse_args_ResNet30FAN():
    '''RMB冠号识别模型参数设置'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/competitions/TinyMind/net/ResNet30Rec/model/CapitalDateRecogLSTM.prototxt')
    parser.add_argument('--model_weights',
                        default='/work/competitions/TinyMind/net/ResNet30Rec/weights/money_rec_iter_1000.caffemodel')
    parser.add_argument('--time_step', default=41)                        
    parser.add_argument('--image_resize', default=[32, 160], type=int)
    parser.add_argument('--mean_value', default=[120, 127, 130])
    parser.add_argument('--input_scale', default=0.00390625)
    parser.add_argument('--resize_type', default='')
    return parser.parse_args()

def parse_args():
    '''RMB冠号识别模型参数设置'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/competitions/TinyMind/net/VaiRec/deploy.prototxt')
    parser.add_argument('--model_weights',
                        default='/work/competitions/TinyMind/net/VaiRec/tinymind_id_iter_6000.caffemodel')
    parser.add_argument('--time_step', default=40)                        
    parser.add_argument('--image_resize', default=[32, 160], type=int)
    parser.add_argument('--mean_value', default=[120, 127, 130])
    parser.add_argument('--input_scale', default=0.00390625)
    parser.add_argument('--resize_type', default='')
    return parser.parse_args()

def refuse2crownNum(image_file, lstm_classification, showkey=False):
    prob, prob_index = lstm_classification.classify(image_file, 'permute_fc') # permute_fc premuted_fc_a
    new_prob = [SoftMax(prob[i]) for i in range(len(prob))]
    pre_result, rrprob, rlist, rprob = get_ctc_decoder_refuse(prob_index, new_prob, RMBmap)
    plate_score = reduce(lambda x, y: x*y, rrprob)
    img_name = os.path.basename(image_file)
    img_item = img_name.split('_')[0]
    img_folder = os.path.basename(os.path.dirname(image_file))
    label = img_name.split('_')[0] if '_' in img_name else ''

    pre_flag = 1 if pre_result == label else 0

    result = 1

    if showkey:
        logging.info("识别结果为:%s[%s], 整合代表概率为: %s \n 具体概率为: %s \n" %(pre_result, len(pre_result), str(Decimal(plate_score).quantize(Decimal('0.01'))), str(' ').join(str(i) for i in rrprob)))
        # logging.info("详细slice分布如下(含空格分数):")
        for i in range(len(rprob)):
            pass
            # logging.info(RMBmap[rlist[i]] + ' ' + str(rprob[i]))
        logging.info("=====================================\n")
        save_ok_dir = '/work/competitions/TinyMind/data/finals/'+ img_folder + '_ok2check'
        if img_item in ok_dict:
            if pre_result == ok_dict[img_item]:
                return result, pre_flag
            else:
                save_name = pre_result + '_' + ok_dict[img_item] + '_' + img_name
                save_path = os.path.join(save_ok_dir, save_name)
                shutil.copy(image_file, save_path)
                return result, pre_flag

        if len(pre_result) != 10:
            save_dir = '/work/competitions/TinyMind/data/finals/'+ img_folder + '_check'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_name = pre_result + '_' + img_name
            save_path = os.path.join(save_dir, save_name)
            shutil.copy(image_file, save_path)
            return result, pre_flag

        if plate_score < 0.9:
            save_dir = '/work/competitions/TinyMind/data/finals/'+ img_folder + '_0.9check'
            save_name = pre_result + '_' + img_item + '_' + str(Decimal(plate_score).quantize(Decimal('0.01'))) + '.jpg'
            save_path = os.path.join(save_dir, save_name)
            shutil.copy(image_file, save_path)
            return result, pre_flag

        save_dir = '/work/competitions/TinyMind/data/finals/'+ img_folder + '_error'
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = pre_result + '_' + img_name
        save_path = os.path.join(save_dir, save_name)
        # if os.path.exists(os.path.join('/work/competitions/TinyMind/data/rec/test/0.5_error', save_name)):
        #     shutil.copy(image_file, save_path)
        shutil.copy(image_file, save_path)
        # img = cv2.imread(image_file)
        # window_name = pre_result + '_' + str(Decimal(plate_score).quantize(Decimal('0.01')))
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        return result, pre_flag
    return result, pre_flag

def crown_number_recognize_lstm(img_dir, showkey=False, _IS_Debug_=False):
    args = parse_args_ResNet30FAN()
    lstm_classification = CaffeLstmClassification(args.gpu_id, args.model_def, args.model_weights, args.time_step,
                                                  args.image_resize, args.mean_value, args.input_scale, args.resize_type)
    
    img_list = os.listdir(img_dir)

    data_size = len(img_list)

    logging.info("Deal with the file: %s[size: %s]" %(img_dir, data_size))
    
    cnt = 0

    pre_cnt = 0
    
    for i, item in enumerate(img_list):
        image_file = os.path.join(img_dir, item)
        isPlate, pre_flag = refuse2crownNum(image_file, lstm_classification, showkey)

        if (isPlate == 1 and pre_flag == 0) and _IS_Debug_:
            logging.info("[%s]误判的图片为: %s" %(str(i), image_file))
            refuse2crownNum(image_file, lstm_classification, True)

        cnt += isPlate
        pre_cnt += pre_flag
        if pre_flag:
            print(image_file)

        if i%1000 == 0:
            logging.info("Deal with [%s]st" %(str(i)))

    logging.info("拒识准确率为: %s" %(str(cnt*1.0/data_size)))
    logging.info("[%s]识别准确率为: %s \n" %(str(pre_cnt), str(pre_cnt*1.0/data_size)))






if __name__ == '__main__':
    # img_root = '/work/hena/ocr/data/CLPR/rec/testdata/weizhang_old'
    # for item in os.listdir(img_root):
    #     img_dir = os.path.join(img_root, item)
    #     plate_recognize_lstm(img_dir, False, False)
    # 
    crown_number_recognize_lstm('/work/competitions/TinyMind/data/finals/resized', True, False)
    # for i in ['100.0', '50.0', '10.0', '5.0', '2.0', '1.0', '0.5', '0.2', '0.1']:
    #     crown_number_recognize_lstm('/work/competitions/TinyMind/data/rec/test/'+ i +'_8k', True, False)
