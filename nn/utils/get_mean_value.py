#/usr/bin/python3
#-*- coding=utf-8 -*-

import cv2
import os
import sys
import numpy as np
import glob

tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from file2list import readTxt


def batch_cal_image_mean(dirpath):
    img_list = glob.glob(dirpath + "/*.png")
    #img_list = readTxt(dirpath)
    num_sum, b_sum, g_sum, r_sum = 1,1,1,1
    sum_w, sum_h = 0, 0
    max_w, max_h = 0, 0
    for item in img_list:
        if not os.path.exists(item):
            print(item)
            continue
        img = cv2.imread(item)
        h, w = img.shape[: 2]
        sum_h += h
        sum_w += w

        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h

        assert(cv2.split(img)), item
        tmp_b_sum, tmp_g_sum, tmp_r_sum = list(map(lambda x: np.sum(x), cv2.split(img)))
        num_sum += img.shape[0]*img.shape[1]
        b_sum += tmp_b_sum
        g_sum += tmp_g_sum
        r_sum += tmp_r_sum
    print("========== image mean value =========")
    print("b    g   r   h   w   maxw    maxh")
    print(list(map(lambda x: x / num_sum, [b_sum, g_sum, r_sum])), sum_h/len(img_list), sum_w/len(img_list), max_w, max_h)
    return list(map(lambda x: x / num_sum, [b_sum, g_sum, r_sum]))

if __name__ == '__main__':
    dirpath = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/Date/CDate/Voucher/20180925/roi'
    batch_cal_image_mean(dirpath)