#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
这个代码主要实现的功能为: 车牌拒识
'''

import os
import sys
import cv2
import math
from functools import reduce
from decimal import Decimal
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
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

CLPRmap = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', '京', '沪', '津', '渝', '黑', '吉',
           '辽', '蒙', '冀', '新', '甘', '青', '陕', '宁', '豫', '鲁',
           '晋', '皖', '鄂', '湘', '苏', '川', '贵', '云', '桂', '藏',
           '浙', '赣', '粤', '闽', '琼', '挂', '学', '警', ' ']


def get_ctc_decoder_refuse(prob_index, prob, maplist, black_id=0):
    if black_id == 0:
        black_id = len(maplist) - 1
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

def parse_args():
    '''车牌识别模型参数设置'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/CLPR/rec/V1.0.1/deploy.prototxt')
    parser.add_argument('--image_resize', default=[112, 240], type=int)
    parser.add_argument('--mean_value', default=[0, 0, 0])
    parser.add_argument('--time_step', default=60)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/CLPR/rec/V1.0.1/original_nocolor_iter_150000.caffemodel')
    return parser.parse_args()

def refuse2plate(image_file, lstm_classification, showkey=False):
    prob, prob_index = lstm_classification.classify(image_file, 'premuted_fc')
    new_prob = [SoftMax(prob[i]) for i in range(len(prob))]
    pre_result, rrprob, rlist, rprob = get_ctc_decoder_refuse(prob_index, new_prob, CLPRmap)
    plate_score = reduce(lambda x, y: x*y, rrprob)
    if showkey:
        result = 0
        logging.info("识别结果为:%s, 整合代表概率为: %s \n 具体概率为: %s \n" %(pre_result, str(plate_score), str(' ').join(str(i) for i in rrprob)))
        logging.info("详细slice分布如下(含空格分数):")
        for i in range(len(rprob)):
            print(CLPRmap[rlist[i]] + ' ' + str(rprob[i]))
        if len(rrprob) < 6 or len(rrprob) > 9 or str(pre_result)[0].isdigit() or plate_score < 0.1:
            logging.info("非有效车牌")
        else:
            logging.info("有效车牌")
            result = 1
        logging.info("=====================================\n")
        
        img = cv2.imread(image_file)
        window_name = pre_result + '_' + str(Decimal(plate_score).quantize(Decimal('0.01')))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        return result
    else:
        if len(rrprob) < 6 or len(rrprob) > 9 or str(pre_result)[0].isdigit() or plate_score < 0.1:
            return 0
        else:
            return 1

def plate_recognize_lstm(img_dir, save_error_dir, showkey=False):
    args = parse_args()
    lstm_classification = CaffeLstmClassification(args.gpu_id, args.model_def, args.model_weights, args.time_step,
                                                  args.image_resize, args.mean_value)
    
    img_list = os.listdir(img_dir)

    data_size = len(img_list)

    logging.info("Deal with the file: %s[size: %s] \n" %(img_dir, data_size))
    
    cnt = 0
    
    for i, item in enumerate(img_list):
        image_file = os.path.join(img_dir, item)
        isPlate = refuse2plate(image_file, lstm_classification, showkey)

        if isPlate == 0:
            logging.info("[%s]误判的图片为: %s" %(str(i), image_file))
            refuse2plate(image_file, lstm_classification, True)

        cnt += isPlate

        if i%1000 == 0:
            logging.info("Deal with [%s]st" %(str(i)))

    logging.info("拒识准确率为: %s" %(str(cnt*1.0/data_size)))






if __name__ == '__main__':
    img_dir = '/work/hena/ocr/data/CLPR/rec/test-data/weizhang_6-12'
    save_error_dir = '/work/hena/ocr/data/CLPR/refuse/refused_error'
    plate_recognize_lstm(img_dir, save_error_dir, False)

