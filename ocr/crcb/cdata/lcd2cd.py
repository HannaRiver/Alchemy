#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import logging
import shutil
import json
import cv2

from lca2ca import get_chn_num

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def json2roi_label(json_path):
    '''
    范围json中的rect及label
    '''
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for item in obj:
            roi_x, roi_y, roi_w, roi_h = [int(i) for i in item['rect']]
            label = item['label']
            if roi_w <= 10 or roi_h <= 10:
                print("ROI Json Warning::", json_path)
            else:
                return roi_x, roi_y, roi_w, roi_h, label

    return roi_x, roi_y, roi_w, roi_h, label

def num2date(value):
    '''
    数字日期转换为中文大写日期
    '''
    assert(len(value) == 8), 'date must len = 8'
    num = ('零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖')

    rstring = ""
    for i, astr in enumerate(value):
        if i == 4:
            rstring += '年'
        if i == 4 and astr == '1':
            rstring += '拾'
            continue
        if i == 6:
            rstring += '月'
        if i == 6 and astr != '0':
            rstring = rstring + num[int(astr)] + '拾'
            continue
        if i == 7 and astr == '0':
            continue
        rstring += num[int(astr)]
    rstring += '日'

    return rstring

def get_document_date(img_path, json_path, save_dir, pre_name, label_value):
    '''
    获取凭证日期roi，将凭证日期分为中文和数字
    '''
    img = cv2.imread(img_path)
    roi_x, roi_y, roi_w, roi_h, label = json2roi_label(json_path)
    roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]

    if label == 'smallCar':
        save_path = os.path.join(save_dir, 'num_roi', pre_name + label_value + '.png')
        save_img_path = os.path.join(save_dir, 'num', pre_name + label_value + '.png')

    else:
        save_path = os.path.join(save_dir, 'chn_roi', pre_name + num2date(label_value) + '.png')
        save_img_path = os.path.join(save_dir, 'chn', pre_name + label_value + '.png')
    
    cv2.imwrite(save_path, roi_img)
    shutil.copyfile(img_path, save_img_path)

def batch_get_roi_CkDate(img_dir, json_dir, save_dir):
    '''
    crcb shrd data 支票签发日期小写日期2大写日期
    '''
    img_name_list = os.listdir(img_dir)

    for item in img_name_list:
        tmp_label = item.split('_')
        pre_label = tmp_label[0] + '_' + tmp_label[1] + '_'
        label_value = tmp_label[2].split('.png')[0]
        img_path = os.path.join(img_dir, item)
        json_path = os.path.join(json_dir, item+'.json')

        save_roi_path = os.path.join(save_dir, pre_label + num2date(label_value) + '.png')

        get_chn_num(img_path, json_path, save_roi_path)

def batch_get_roi_DocDate(img_dir, json_dir, save_dir):
    '''
    批量处理获取凭证日期roi并分类
    '''
    img_name_list = os.listdir(img_dir)

    for item in img_name_list:
        tmp_label = item.split('_')
        pre_label = tmp_label[0] + '_' + tmp_label[1] + '_'
        label_value = tmp_label[2].split('.png')[0]
        img_path = os.path.join(img_dir, item)
        json_path = os.path.join(json_dir, item+'.json')

        get_document_date(img_path, json_path, save_dir, pre_label, label_value) 

def RenameCkDate(img_root, date_key):
    '''
    对支票签发日期重命名
    '''
    item_name = '支票签发日期'
    img_dir = os.path.join(img_root, item_name, date_key, 'mark')
    save_dir = os.path.join(img_root, item_name, date_key, 'roi')
    json_dir = os.path.join(img_root, item_name, date_key, 'json')

    logging.info("Deal with the folder: %s" %(img_dir))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    batch_get_roi_CkDate(img_dir, json_dir, save_dir)

def RenameDocDate(img_root, date_key):
    '''
    对凭证日期转换标签
    '''
    item_name = '凭证日期'
    img_dir = os.path.join(img_root, item_name, date_key, 'mark')
    json_dir = os.path.join(img_root, item_name, date_key, 'json')
    save_dir = os.path.join(img_root, item_name, date_key)

    os.mkdir(os.path.join(save_dir, 'num'))
    os.mkdir(os.path.join(save_dir, 'chn'))
    os.mkdir(os.path.join(save_dir, 'num_roi'))
    os.mkdir(os.path.join(save_dir, 'chn_roi'))


    logging.info("Deal with the folder: %s" %(save_dir))

    batch_get_roi_DocDate(img_dir, json_dir, save_dir)



if __name__ == '__main__':
    date_key = '20180731'
    img_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/Date'

    # RenameCkDate(img_root, date_key)
    
    RenameDocDate(img_root, date_key)

