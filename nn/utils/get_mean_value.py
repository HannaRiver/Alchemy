#/usr/bin/python3
#-*- coding=utf-8 -*-

import cv2
import os
import sys
import numpy as np
import glob
import shutil
from matplotlib import pyplot as plt

tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from file2list import readTxt


def DrawHist(lenths):
    '''
    lenths: np.array([])
    '''
    pass


def batch_cal_image_mean(dirpath):
    img_list = glob.glob(dirpath + "/*/*.jpg")
    # img_list = readTxt(dirpath)
    num_sum, b_sum, g_sum, r_sum = 1,1,1,1
    sum_w, sum_h = 0, 0
    max_w, max_h = 0, 0
    label_max = 0
    h_list, w_list, h_w_list = [], [], []
    label_list = []
    for item in img_list:
        if not os.path.exists(item):
            print(item)
            continue
        img = cv2.imread(item)
        if img is None:
            print("Img Empty: %s" %item)
            os.remove(item)
            continue
        img_name = os.path.basename(item)
        label = os.path.splitext(img_name)[0].split('_')[-1]
        # label_max = len(label) if len(label) > label_max else label_max
        label_list.append(len(label))
        h, w = img.shape[: 2]
        h_list.append(h)
        w_list.append(w)
        h_w_list.append(float(h) / w)
        sum_h += h
        sum_w += w
        if float(w) / h > 20:
            pass
            # shutil.move(item, os.path.join('/work/competitions/ICDAR/SROIE/task2/tmp', img_name))
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h

        assert(cv2.split(img)), item
        tmp_b_sum, tmp_g_sum, tmp_r_sum = list(map(lambda x: np.sum(x), cv2.split(img)))
        num_sum += img.shape[0] * img.shape[1]
        b_sum += tmp_b_sum
        g_sum += tmp_g_sum
        r_sum += tmp_r_sum
    print("========== image mean value =========")
    print("b    g   r   h   w   maxw    maxh")
    print(list(map(lambda x: x / num_sum, [b_sum, g_sum, r_sum])), sum_h/len(img_list), sum_w/len(img_list), max_w, max_h)
    label_max = max(label_list)
    label_max_value = img_list[label_list.index(label_max)]
    
    print(label_max, label_max_value)
    ################ 画频次分布直方图 ###################
    h_vec = np.array(h_list)
    w_vec = np.array(w_list)
    ratio_vec = np.array(h_w_list)
    kwargs = dict(histtype='stepfilled', alpha=0.3, normed=False, bins=40)
    # plt.hist(h_vec, **kwargs)
    # plt.hist(w_vec, **kwargs)
    # plt.hist(ratio_vec, **kwargs)
    # plt.show()

    return list(map(lambda x: x / num_sum, [b_sum, g_sum, r_sum]))


if __name__ == '__main__':
    dirpath = '/work/competitions/TinyMind/PreTrainData'
    batch_cal_image_mean(dirpath)