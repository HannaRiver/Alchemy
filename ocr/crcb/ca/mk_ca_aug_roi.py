#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import json
import cv2
import logging
import random
import sys
import shutil
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'nn', 'utils'))

from mk_resize_data import resize_img


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
from json2roi import json2roi


def batch_get_aug_roi(img_dir, json_dir, save_dir, date):
    img_name_list = os.listdir(img_dir)
    for item in img_name_list:
        img_path = os.path.join(img_dir, item)
        json_path = os.path.join(json_dir, item+'.json')

        roi_x, roi_y, roi_w, roi_h = json2roi(json_path)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[: 2]
        for i in range(10):
            # random_x1 = int(roi_h/4 * random.random())
            # random_x2 = int(5 * random.random())
            random_x1 = int(5 * random.random())
            random_x2 = int(5 * random.random())
            random_y1 = int(5 * random.random())
            random_y2 = int(5 * random.random())

            if i == 0 and False:
                random_x1, random_x2, random_y1, random_y2 = 0, 0, 0, 0

            roi_x = max(roi_x - random_x1, 0)
            roi_y = max(roi_y - random_y1, 0)
            roi_w = min(roi_w + random_x1 + random_x2, img_w-roi_x)
            roi_h = min(roi_h + random_y2 + random_y1, img_h-roi_y)
            key = date + '_' + str(random_x1) + str(random_x2) + str(random_y1) + str(random_y1) + '_'

            save_roi_path = os.path.join(save_dir, key + item)

            roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]

            # cv2.imwrite(save_roi_path, resize_img(roi_img, 48, 240, [150, 139, 138]))
            cv2.imwrite(save_roi_path, roi_img)



def mk_ca_aug_roi():
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/CA'
    item_list = ['CA', 'Ck', 'DR']
    date_list = ['20180731', '20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']

    for item in item_list:
        for date in date_list:
            img_dir = os.path.join(data_root, item, 'labeled', date, 'mark')
            json_dir = os.path.join(data_root, item, 'labeled', date, 'json')
            save_dir = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/CA_aug_data'
            # shutil.rmtree(save_dir)

            logging.info("Deal with the folder: %s" %(img_dir))

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            batch_get_aug_roi(img_dir, json_dir, save_dir, date)

def tmp_function():
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/CA'
    item_list = ['CA', 'Ck', 'DR']
    date_list = ['20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']
    fw = open('/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/data/V0.0.3/traindata_new.txt', 'w')
    for item in item_list:
        for date in date_list:
            save_dir = os.path.join(data_root, item, date, 'mark_aug_roi_1')

            logging.info("Deal with the folder: %s" %(save_dir))

            if not os.path.exists(save_dir):
                continue

            img_list = os.listdir(save_dir)

            for img_name in img_list:
                img_path = os.path.join(save_dir, img_name)
                fw.write(img_path + '\n')
    fw.close()
if __name__ == '__main__':
    mk_ca_aug_roi()
    # tmp_function()