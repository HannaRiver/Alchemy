#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import re
import cv2
import numpy as np
from file2list import readTxt


def undeform_resize(img, resize_h, resize_w, mean_value=[150, 139, 138]):
    '''
    不变形填背景resize
    '''
    new_img = cv2.merge([np.ones((resize_h, resize_w)) * i for i in mean_value])
    new_img = new_img.astype(img.dtype)
    img_h, img_w = img.shape[0], img.shape[1]
    ratio_h = float(resize_h) / img_h
    ratio_w = float(resize_w) / img_w
    if ratio_h < ratio_w:
        # new_img[:, :ratio_h * img_w] = cv2.resize(resize_h, ratio_h * img_w)
        new_img[:, :int(ratio_h * img_w)] = cv2.resize(img, (int(ratio_h * img_w), resize_h), interpolation=cv2.INTER_CUBIC)
    else:
        # new_img[: ratio_w * img_h, :] = img.resize(ratio_w * img_h, resize_w)
        new_img[: int(ratio_w * img_h), :] = cv2.resize(img, (resize_w, int(ratio_w * img_h)), interpolation=cv2.INTER_CUBIC)

    return new_img

def undeform_center_resize(img, resize_h, resize_w, mean_value=[145.0, 140.0, 144.0]):
    '''
    不变形填背景图像放中间resize
    '''
    new_img = cv2.merge([np.ones((resize_h, resize_w)) * i for i in mean_value]).astype(img.dtype)
    new_img = new_img.astype(img.dtype)
    img_h, img_w = img.shape[0], img.shape[1]
    ratio_h = float(resize_h) / img_h
    ratio_w = float(resize_w) / img_w
    if ratio_h < ratio_w:
        tmp_img = cv2.resize(img, (int(ratio_h * img_w), resize_h), interpolation=cv2.INTER_CUBIC)
        star_point = int((resize_w - ratio_h * img_w)/2)
        new_img[:, star_point: star_point + tmp_img.shape[1]] = tmp_img
    else:
        tmp_img = cv2.resize(img, (resize_w, int(ratio_w * img_h)), interpolation=cv2.INTER_CUBIC)
        star_point = int((resize_h - ratio_w * img_h)/2)
        new_img[star_point: star_point + tmp_img.shape[0], :] = tmp_img
    
    return new_img 

def resize_with_center_pad(img, resize_h, resize_w, mean_value=[0, 0, 0]):
    '''
    Gao Chao way to resize img for lstm net
    '''
    img_h, img_w = img.shape[: 2]
    mean_value = [np.mean(img[:,:,i]) for i in [0, 1, 2]] if mean_value == [0, 0, 0] else mean_value
    assert(img_w*img_h*resize_w*resize_h != 0), 'w, h have 0 value!'
    orig_aspect, new_aspect = img_w*1.0 / img_h, resize_w*1.0 / resize_h

    if orig_aspect > new_aspect:
        img_resized  = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    else:
        width = int(orig_aspect * resize_h)
        img_resized = cv2.resize(img, (width, resize_h), interpolation=cv2.INTER_LINEAR)
        resSize = img_resized.shape[: 2]
        padding_l = int(round((resize_w - resSize[1]) * 0.5))
        padding_r = resize_w - resSize[1] - padding_l
        img_resized = cv2.copyMakeBorder(img_resized, 0, 0, padding_l, padding_r,
                                         cv2.BORDER_CONSTANT, value=mean_value)
    
    return img_resized

def resize_with_height_pad(img, resize_h, resize_w, mean_value=[0, 0, 0]):
    '''
    He Na way to resize img for lstm net
    '''
    img_h, img_w = img.shape[: 2]
    mean_value = [np.mean(img[:,:,i]) for i in [0, 1, 2]] if mean_value == [0, 0, 0] else mean_value
    assert(img_w*img_h*resize_w*resize_h != 0), 'w, h have 0 value!'
    orig_aspect, new_aspect = img_w*1.0 / img_h, resize_w*1.0 / resize_h

    if orig_aspect > new_aspect:
        img_resized  = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    else:
        width = int(orig_aspect * resize_h)
        img_resized = cv2.resize(img, (width, resize_h), interpolation=cv2.INTER_LINEAR)
        resSize = img_resized.shape[: 2]
        padding_l = 0
        padding_r = resize_w - resSize[1]
        img_resized = cv2.copyMakeBorder(img_resized, 0, 0, padding_l, padding_r,
                                         cv2.BORDER_CONSTANT, value=mean_value)
    
    return img_resized

def map4undeform_center_resize(point, resize_size, original_size):
    '''
    input: undeform_center_resize 下的坐标[x, y]
    resize_size = [H, W]; original_size = [h, w]
    H: resize height; W: resized width; h: original height; w: original width
    return: 原始图片的对应坐标 [x', y']
    '''
    H, W = resize_size
    h, w = original_size
    assert(H*W*h*w != 0), 'map4undeform_center_resize error:: H, W, h, w can not be 0!'
    assert(len(point) == 2), 'map4undeform_center_resize error:: [x, y] v.s. ' + str(point)
    ratio_h, ratio_w = float(H)/h, float(W)/w
    if ratio_h < ratio_w:
        x = min(max(int(point[0]/ratio_h + w/2.0 - W/(ratio_h*2)), 0), w)
        y = min(max(int(point[1]/ratio_h), 0), h)
    else:
        x = min(max(int(point[0]/ratio_w), 0), w)
        y = min(max(int(point[1]/ratio_w + h/2.0 - H/(ratio_w*2)), 0), h)
    
    return x, y

def batch_resize_img(imgtxt, resize_h, resize_w, mean_value, save_dir):
    file_list = readTxt(imgtxt)

    for item in file_list:
        # item = os.path.join(imgtxt, item)
        if not os.path.exists(item):
            print("Error: file not exists: ", item)
            continue

        img = cv2.imread(item)

        img_name = re.split(r'\/', item)[-1]
        date = re.split(r'\/', item)[-3] + '_' + re.split(r'\/', item)[-2]
        img_id = img_name.split('_')[0]
        re_name = img_name[len(img_id): ]

        tmp_id = int(img_id)

        save_path = os.path.join(save_dir, date + '_' + img_name)

        while os.path.exists(save_path):
            tmp_id += 1
            save_path = os.path.join(save_dir, date + '_' + str(tmp_id) + re_name)
        
        cv2.imwrite(save_path, undeform_resize(img, resize_h, resize_w, mean_value))