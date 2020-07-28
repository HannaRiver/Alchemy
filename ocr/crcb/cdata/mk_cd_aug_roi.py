#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import json
import cv2
import random
import sys
import shutil
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from resize_img import undeform_resize
from file2list import readTxt
from json2roi import json2roi


def get_aug_roi(im_path, json_path, date, save_dir):
    roi_x, roi_y, roi_w, roi_h = json2roi(json_path)
    img = cv2.imread(im_path)
    img_name = os.path.basename(im_path)
    img_h, img_w = img.shape[: 2]

    for i in range(10):
            # random_x1 = int(roi_h/4 * random.random())
            # random_x2 = int(5 * random.random())
        random_x1 = int(5 * random.random())
        random_x2 = int(5 * random.random())
        random_y1 = int(5 * random.random())
        random_y2 = int(5 * random.random())


        roi_x = max(roi_x - random_x1, 0)
        roi_y = max(roi_y - random_y1, 0)
        roi_w = min(roi_w + random_x1 + random_x2, img_w-roi_x)
        roi_h = min(roi_h + random_y2 + random_y1, img_h-roi_y)
        
        key = date + '_' + str(random_x1) + str(random_x2) + str(random_y1) + str(random_y1) + '_'

        save_roi_path = os.path.join(save_dir, key + img_name)

        roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]

        cv2.imwrite(save_roi_path, undeform_resize(roi_img, 32, 240, [143, 132, 125]))

def mk_cd_aug_roi():
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/Date/CDate'
    img_list_path = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_date/data/V0.0.1/train_org.txt'
    save_dir = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/Train/Cdate/cd_rec_v002_train_aug'
    date_list = ['20180731',
                 '20180802', '20180803', '20180808', '20180813', '20180814',
                 '20180816', '20180827', '20180828', '20180829']
    for img_path in readTxt(img_list_path):
        img_name = os.path.basename(img_path)
        date = img_path.split('/')[-3]

        if date not in date_list:
            continue

        im_path = os.path.abspath((os.path.join(os.path.dirname(img_path), '..', 'mark', img_name)))
        json_path = os.path.abspath((os.path.join(os.path.dirname(img_path), '..', 'json', img_name + '.json')))

        if not os.path.exists(im_path) or not os.path.exists(json_path):
            print("%s Can not make aug date, file not exists" %(img_path))
            continue
        
        get_aug_roi(im_path, json_path, date, save_dir)
        
if __name__ == '__main__':
    mk_cd_aug_roi()
    