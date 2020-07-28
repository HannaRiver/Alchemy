#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
这个代码主要实现的功能为: 碎片识别模块
'''
import os
import sys
import cv2
import math
from functools import reduce
from decimal import Decimal
from alchemy_config import cfg
tool_root = cfg.ROOT_DIR
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))
sys.path.append(os.path.join(tool_root, 'utils'))
os.environ['GLOG_minloglevel'] = '2'
from lstm import CaffeLstmClassification
from file2list import readTxt, load_wordslib
import pickle
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


DictMap = {
    'CA': ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
         '拾', '元', '佰', '仟', '万', '亿', '角', '分', '整', '正',
         '圆', ' '],
    'CD': ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
          '拾', '年', '月', '日', ' '],
    'Date': [' ', '零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
          '拾', '年', '月', '日', '0', '1', '2', '3', '4', '5', '6', '7',
          '8', '9'],
    'LA': ["0","1","2","3","4","5","6","7","8","9", "￥", " "],
    # 'Chn': load_wordslib(cfg.CRCB_WORDSLIB.SHRED_CHN)
    'Chn': []
}

def get_recognition(args):
    # args = parse_args()
    classification = CaffeLstmClassification(args.gpu_id, args.model_def, args.model_weights, 
                                             args.time_step, args.image_resize, args.mean_value, 
                                             args.scale, args.resize_type)
    return classification

def SoftMax(net_ans):
    tmp_net = [math.exp(i) for i in net_ans]
    sum_exp = sum(tmp_net)
    return [i/sum_exp for i in tmp_net]

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

def refuse2plate(image_file, lstm_classification, rec_item='CA', showkey=False, label=''):
    CLPRmap = DictMap[rec_item]
    prob, prob_index = lstm_classification.classify(image_file, 'premuted_fc')
    new_prob = [SoftMax(prob[i]) for i in range(len(prob))]
    pre_result, rrprob, rlist, rprob = get_ctc_decoder_refuse(prob_index, new_prob, CLPRmap)
    plate_score = reduce(lambda x, y: x*y, rrprob) if pre_result else rprob[0]
    img_name = os.path.basename(image_file) if type(image_file) == type('1024') else '传的是图片啊!'
    label = img_name.split('_')[0] if '_' in img_name else '' if not label else label

    pre_flag = 1 if pre_result[-len(label): ] == label else 0

    result = 1

    if showkey:
        logging.info("识别结果为:%s[%s], 整合代表概率为: %s \n 具体概率为: %s \n" %(pre_result, label, str(Decimal(plate_score).quantize(Decimal('0.01'))), str(' ').join(str(i) for i in rrprob)))
        logging.info("详细slice分布如下(含空格分数):")
        for i in range(len(rprob)):
            logging.info(CLPRmap[rlist[i]] + ' ' + str(rprob[i]))
        if len(rrprob) < 6 or len(rrprob) > 9 or str(pre_result)[0].isdigit() or plate_score < 0.3:
            logging.info("非有效")
            result = 0
        else:
            logging.info("有效")
        logging.info("=====================================\n")
        
        img = cv2.imread(image_file) if type(image_file) == type('1024') else image_file
        window_name = pre_result + '_' + str(Decimal(plate_score).quantize(Decimal('0.01')))
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", img)
        cv2.waitKey(0)
        return pre_result, result, pre_flag
    else:
        if len(rrprob) < 6 or len(rrprob) > 9 or str(pre_result)[0].isdigit() or plate_score < 0.1:
            result = 0
    return pre_result, result, pre_flag

def parse_args():
    '''车牌识别模型参数设置'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/ocr/clpr/model/rec/V1.0.1/model/V1.0.1.2/deploy.prototxt')
    parser.add_argument('--image_resize', default=[112, 240], type=int)
    parser.add_argument('--mean_value', default=[0, 0, 0])
    parser.add_argument('--time_step', default=60)
    parser.add_argument('--model_weights',
                        default='/work/ocr/clpr/model/rec/V1.0.1/model/V1.0.1.4/clpr_rec_v10110_iter_462000.caffemodel')
    return parser.parse_args()