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
from classify_class import CaffeClassification
from chars_segment import get_chars_bnboxs
from file2list import txt2info
from file2list import get_label4name
from img_aug import ImgAdaptiveBinary


def SoftMax(net_ans):
    tmp_net = [math.exp(i) for i in net_ans]
    sum_exp = sum(tmp_net)
    return [i/sum_exp for i in tmp_net]

def get_prob_label(img_path, probs, index_probs, args, _IS_DEBUG_=False):
    nan_str_list = txt2info(args.hard_list, 1, ' ')
    nan_label_list = txt2info(args.hard_list, 0, ' ')
    yi_str_list = txt2info(args.easy_list, 1, ' ')
    yi_label_list = txt2info(args.easy_list, 0, ' ')
    label_list = txt2info(args.label_list, 0, ' ')

    # easy/hard 对于所有类label中的index
    easy_word_index, hard_word_index = int(yi_str_list[index_probs[2]]), int(nan_str_list[index_probs[3]])   
    new_prob = [SoftMax(probs[i]) for i in range(len(probs))]
    cls_easy_prob, cls_hard_prob = new_prob[0] # easy/hard的分类概率
    easy_word_prob, hard_word_prob = [new_prob[i][index_probs[i]] for i in [2, 3]]
    easy_word_combi_prob, hard_word_combi_prob = cls_easy_prob*easy_word_prob, cls_hard_prob*hard_word_prob  #联合概率
    prob_char = label_list[easy_word_index if easy_word_combi_prob >= hard_word_combi_prob else hard_word_index]
    prob = easy_word_combi_prob if easy_word_combi_prob >= hard_word_combi_prob else hard_word_combi_prob

    easy_word_in_all_cls_prob, hard_word_in_all_cls_prob = [new_prob[1][i] for i in [easy_word_index, hard_word_index]]

    if _IS_DEBUG_:
        if type(img_path) == type('1024'):
            logging.info("img path: %s" %(img_path))
        logging.info("模型预测这个字为 %s 类(prob: %.3f)" %(['easy', 'hard'][index_probs[0]], [cls_easy_prob, cls_hard_prob][int(index_probs[0])]))
        logging.info("在所有类分支中预测的结果为: %s(prob: %.3f)" %(label_list[int(index_probs[1])], new_prob[1][int(index_probs[1])]))
        logging.info("在easy分支中预测的结果为: %s(prob-easy: %.3f, prob-combi: %.3f, prob-all: %.3f)" 
                   %(label_list[easy_word_index], easy_word_prob, easy_word_combi_prob, easy_word_in_all_cls_prob))
        logging.info("在hard分支中预测的结果为: %s(prob-easy: %.3f, prob-combi: %.3f, prob-all: %.3f)" 
                   %(label_list[hard_word_index], hard_word_prob, hard_word_combi_prob, hard_word_in_all_cls_prob))
        logging.info("最终模型预测的结果为: %s(prob: %.3f)" %(prob_char, prob))

        if type(img_path) == type('hello'):
            img = cv2.imread(img_path)
        else:
            img = img_path
        cv2.namedWindow("char_ori_img", cv2.WINDOW_NORMAL)
        cv2.imshow("char_ori_img", img)
        if args.resize_type == 'adaptiveBinary':
            img = cv2.resize(img, (args.image_resize[1], args.image_resize[0]), interpolation=cv2.INTER_CUBIC)
            binary_img = ImgAdaptiveBinary(img)
            cv2.namedWindow("char_binary_img", cv2.WINDOW_NORMAL)
            cv2.imshow("char_binary_img", binary_img)
        cv2.waitKey(0)
        logging.info("====================")

    return prob_char, prob

def chars_recognise(img_path, classification, args, _IS_DEBUG_=False):
    probs, index_probs = classification.get_some_layer(img_path, ['fc3_all', 'fc3_all_lei', 'fc3_yi', 'fc3_nan'])
    prob_char, prob = get_prob_label(img_path, probs, index_probs, args, _IS_DEBUG_)

    return prob_char, prob, probs, index_probs

def parse_args():
    '''
    szp debug model argument
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='', type=str, help='需要分类的图片根目录')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--mean_value', default=[127.5, 127.5, 127.5])
    parser.add_argument('--scale', default=0.0078125)
    parser.add_argument('--resize_type', default='adaptiveBinary', help='图像预处理方式')
    parser.add_argument('--hard_list', default='/work/help/szp/chars/model/nan.txt')
    parser.add_argument('--easy_list', default='/work/help/szp/chars/model/yi.txt')
    parser.add_argument('--label_list', default='/work/help/szp/chars/model/label.txt')
    parser.add_argument('--model_def',
                        default='/work/help/szp/chars/model/deploy_train_zifu.prototxt')
    parser.add_argument('--image_resize', default=[24, 24], type=int)
    parser.add_argument('--model_weights',
                        default='/work/help/szp/chars/model/model_iter_36000.caffemodel')
    return parser.parse_args()

def get_classification(args):
    # args = parse_args()
    classification = CaffeClassification(args.gpu_id, args.model_def, args.model_weights, args.image_resize, args.mean_value, args.scale, (2, 1, 0), args.resize_type)
    return classification

def main(img_dir):
    args = parse_args()    
    classification = CaffeClassification(args.gpu_id, args.model_def, args.model_weights, args.image_resize, args.mean_value, args.scale, (2, 1, 0), args.resize_type)

    cnt, cnt_r = 0, 0
    for item in os.listdir(img_dir):
        img_path = os.path.join(img_dir, item)
        prob_char, prob, probs, index_probs = chars_recognise(img_path, classification, args, True)
        label = get_label4name(item, 2, '_')
        cnt += 1
        if label == prob_char:
            cnt_r += 1
    print(cnt_r*1.0/cnt)

if __name__ == '__main__':
    # main()
    img_dir = '/work/ocr/card/vehicle_license/data/test/cut/3Owner/test/tmp2'
    print(img_dir)
    main(img_dir)