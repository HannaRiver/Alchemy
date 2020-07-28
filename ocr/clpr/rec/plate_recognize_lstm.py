#/usr/bin/python
#-*- encoding=utf-8 -*-

import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
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
import cv2


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
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/CLPR/rec/V1.0.1/model/V1.0.1.2/deploy.prototxt')
    parser.add_argument('--image_resize', default=[112, 240], type=int)
    parser.add_argument('--mean_value', default=[0, 0, 0])
    parser.add_argument('--time_step', default=60)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/CLPR/rec/V1.0.1/model/V1.0.1.4/clpr_rec_v1017_iter_27000.caffemodel')
    return parser.parse_args()

def del_chn(text):
    # "去除字符串中的中文"
    return ''.join([i if ord(i) < 128 else '' for i in text])

def plate_recognize_lstm(img_dir, labelindex=0, save_error_dir=''):
    args = parse_args()
    lstm_classification = CaffeLstmClassification(args.gpu_id, args.model_def, args.model_weights, args.time_step,
                                                  args.image_resize, args.mean_value)
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
            os.mkdir(os.path.join(save_error_dir, 'chn_error'))
            os.mkdir(os.path.join(save_error_dir, 'all_error'))
        else:
            if not os.path.exists(os.path.join(save_error_dir, 'chn_error')):
                os.mkdir(os.path.join(save_error_dir, 'chn_error'))
                os.mkdir(os.path.join(save_error_dir, 'all_error'))
        logging.info("错误图片将会保存，保存地址为: %s\n 中文错误其他都对的文件夹: %s \t 都错: %s" %(save_error_dir, 'chn_error', 'all_error'))

    del_chn_cnt, right_cnt, cnt = 0, 0, 0
    
    for item in img_list:
        cnt += 1
        image_file = os.path.join(img_dir, item)
        if IsFile:
            image_file = item
        save_pre = os.path.basename(item).split('.jpg')[0]
        label = save_pre.split('_')[labelindex]
        _, prob_index = lstm_classification.classify(image_file, 'premuted_fc')
        pro_label = get_ctc_decoder(prob_index, CLPRmap)

        if del_chn(label) == del_chn(pro_label):
            del_chn_cnt += 1
            if label == pro_label:
                right_cnt += 1
            else:
                if IsSave:
                    save_path = os.path.join(save_error_dir, 'chn_error', pro_label + '_' + item)
                    shutil.copy(image_file, save_path)
                # print("[%s]%s,%s,%s" %(cnt, image_file, label, pro_label))
        else:
            if IsSave:
                save_path = os.path.join(save_error_dir, 'all_error', pro_label + '_' + item)
                shutil.copy(image_file, save_path)
            # print("[%s]%s,%s,%s" %(cnt, image_file, label, pro_label))
    print("all cont: %s, right num: %s, del chn right num: %s\n acc: %s, del_chn_acc: %s \n" %(cnt, right_cnt, del_chn_cnt, right_cnt*1.0/cnt, del_chn_cnt*1.0/cnt))

def vir_oldtestset():
    """
    违章最开始的测试集，但是这里的测试按照车牌类别进行测试
    """
    img_root = '/work/ocr/clpr/data/vir/testset/test_old_data'
    map2color = ['白', '单黄', '双黄', '蓝', '新能源']
    logging.info("====== Deal with the Vir old test set =====")
    logging.info("test set dir: %s\n" %(img_root))
    for i in range(5):
        logging.info("======>>> 正在处理%s牌" %(map2color[i]))
        img_txt = os.path.join(img_root, str(i) + '.txt')
        plate_recognize_lstm(img_txt, 1, '')
    logging.info("============================================\n")

def vai_double_yellow_testset():
    img_root = '/work/ocr/clpr/data/vai/testset/rec/double_yellow'
    logging.info("====== Deal with the Vai double yellow test set =====")
    for item in os.listdir(img_root):
        img_dir = os.path.join(img_root, item)
        plate_recognize_lstm(img_dir, 1, '')
    logging.info("============================================\n")

def vai_new_energy_testset():
    img_root = '/work/ocr/clpr/data/vai/testset/rec/new_energy'
    logging.info("====== Deal with the Vai new energy test set =====")
    for item in os.listdir(img_root):
        img_dir = os.path.join(img_root, item)
        plate_recognize_lstm(img_dir, 1, '')
    logging.info("============================================\n")

def vai_emtest_testset():
    img_root = '/work/ocr/clpr/data/vai/testset/rec/emtest'
    logging.info("====== Deal with the Vai Emtest test set =====")
    for item in os.listdir(img_root):
        img_dir = os.path.join(img_root, item)
        plate_recognize_lstm(img_dir, 0, '')
    logging.info("============================================\n")

def vir_emtest_clear_testset():
    img_dir = '/work/ocr/clpr/data/vir/testset/plate_det_qingxi'
    logging.info("====== Deal with the Vir Emtest clear test set =====")
    plate_recognize_lstm(img_dir, 0, '')
    logging.info("============================================\n")



if __name__ == '__main__':
    vir_oldtestset()
    vai_double_yellow_testset()
    vai_emtest_testset()
    vai_new_energy_testset()
    pass

