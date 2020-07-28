#/usr/bin/python
#-*- encoding=utf-8 -*-

import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'classification'))

import argparse
import shutil
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

import cv2
from lstm import CaffeLstmClassification
from lstm import get_ctc_decoder_CA
from lstm import CAmatch
from lca2ca import cncurrency
from lstm import stdCA


def main(data_root, item_list, date):
    map_list = ['CA', 'Ck', 'DR']
    lstmargs = lstm_args()
    lstm_classification = CaffeLstmClassification(lstmargs.gpu_id,
                                                  lstmargs.model_def, lstmargs.model_weights,
                                                  lstmargs.time_step,
                                                  lstmargs.image_resize)

    for i, item in enumerate(item_list):
        org_dir = os.path.join(data_root, item, date)
        data_dir = os.path.join(data_root, item, date+'ssd_result')
        save_dir = os.path.join(data_root, item, date+'check_result')

        right_dir = os.path.join(data_root, item, date+'right')

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if not os.path.exists(right_dir):
            os.mkdir(right_dir)

        right_save_dir = os.path.join(data_root, 'CA', map_list[i], date)
        if not os.path.exists(right_save_dir) and False:
            os.mkdir(right_save_dir)
            os.mkdir(os.path.join(right_save_dir, 'ssd_roi'))
        
        
        img_name_list = os.listdir(data_dir)
        logging.info("Deal with the file: ./%s/%s[size: %s] \n" %(item, date, len(img_name_list)))

        cnt, right = 0, 0
        for img_name in img_name_list:
            image_file = os.path.join(data_dir, img_name)
            if not os.path.exists(image_file):
                continue
            save_path = os.path.join(save_dir, img_name)

            tmp_label = img_name.split('_')
            pre_label = tmp_label[0] + '_' + tmp_label[1] + '_'
            label = img_name.split('_')[-1][: -4]
            if label == '' and True:
                continue
            cnt += 1
            _, prob_index = lstm_classification.classify(image_file)
            pro_label = get_ctc_decoder_CA(prob_index)
            ca_label = cncurrency(label)
            # ca_label = label

            if CAmatch(ca_label, pro_label) or False:
                right += 1
                new_img_name = pre_label + pro_label + '.png'
                # shutil.move(image_file, os.path.join(right_save_dir, 'ssd_roi', new_img_name))
                if not os.path.exists(os.path.join(org_dir, img_name)):
                    print(os.path.join(org_dir, img_name))
                    continue

                shutil.move(os.path.join(org_dir, img_name), os.path.join(right_dir, new_img_name))
            else:
                # logging.info("imgname:%s label:%s[%s] --> %s" %(img_name, label, cncurrency(label), pro_label))
                # shutil.copy(os.path.join(org_dir, img_name), save_path)
                if not os.path.exists(os.path.join(org_dir, img_name)):
                    print(os.path.join(org_dir, img_name))
                    continue
                shutil.move(os.path.join(org_dir, img_name), os.path.join(org_dir, pre_label + ca_label + '.png'))
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # cv2.imwrite(save_path, img)
        logging.info("%s-acc: %s \n" %(item, right*1.0/cnt))

def lstm_args_V003():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/CA/CA/20180731/roi/0_大写金额_捌佰肆拾叁元整.png')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/model/V0.0.3/ca.prototxt')
    parser.add_argument('--image_resize_h', default=32, type=int)
    parser.add_argument('--image_resize_w', default=160, type=int)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/weights/V0.0.3_0925/ca_iter_970000.caffemodel')
    return parser.parse_args()

def lstm_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/CA/CA/20180731/roi/0_大写金额_捌佰肆拾叁元整.png')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/model/V0.1.1/ca_v011_deploy.prototxt')
    parser.add_argument('--image_resize_h', default=48, type=int)
    parser.add_argument('--image_resize_w', default=240, type=int)
    parser.add_argument('--time_step', default=60)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/weights/V0.1.1/ca_iter_V010_10W.caffemodel')
    return parser.parse_args()

if __name__ == '__main__':
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item'
    item_list = ['大写金额', '支票大写金额', '进账单大写金额']
    date = '20181029'
    main(data_root, item_list, date)
