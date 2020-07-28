#/usr/bin/python3
#-*- coding=utf-8 -*-

import cv2
from file2list import readTxt

def txt2roi(txt_path):
    '''
    违章标签保存格式为x y w h, 默认只有一个目标
    '''
    rect_info = readTxt(txt_path)
    rect = [int(i) for i in rect_info[0].split(' ')]

    assert(len(rect) == 4), "不是正确的Rect: " + str(rect)
    return rect

def get_car_obj(img_path, txt_path, save_roi_path):
    '''
    保存车身roi
    '''
    img = cv2.imread(img_path)
    roi_x, roi_y, roi_w, roi_h = txt2roi(txt_path)
    roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
    cv2.imwrite(save_roi_path, roi_img)