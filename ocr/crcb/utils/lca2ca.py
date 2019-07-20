#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import sys
import json
import cv2
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
from alchemy_config import cfg
sys.path.append(cfg.UTILS_DIR)
from json2roi import get_chn_num
from decimal import Decimal


def cncurrency(value):
    '''
    人民币小写转大写
    '''
    dunit = ('角', '分')
    num = ('零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖')
    iunit = ['元', '拾', '佰', '仟', '万', '拾', '佰', '仟','亿', '拾', '佰', '仟', '万', '拾', '佰', '仟']
    
    if len(value) == 0 or not value[0].isdigit():
        return value


    if not isinstance(value, Decimal):
        value = Decimal(value).quantize(Decimal('0.01'))

    s = str(value)
    assert(len(s) < 19), value + ' is too long ~'
    istr, dstr = s.split('.')
    istr = istr[::-1]
    so = []

    if value == 0:
        return num[0] + iunit[0]
    haszero = False
    if dstr == '00':
        haszero = True
    
    # 处理小数部分
    # 分
    if dstr[1] != '0':
        so.append(dunit[1])
        so.append(num[int(dstr[1])])
    else:
        so.append('整')
    
    # 角
    if dstr[0] != '0':
        so.append(dunit[0])
        so.append(num[int(dstr[0])])
    elif dstr[1] != '0':
        so.append(num[0])
        haszero = True
    
    # 无整数部分
    if istr == '0':
        if haszero:
            so.pop()
        so.reverse()
        return ''.join(so)
    
    # 处理整数部分
    for i, n in enumerate(istr):
        n = int(n)
        if i % 4 == 0:
            if i == 8 and so[-1] == iunit[4]:
                so.pop()
            so.append(iunit[i])
            if n == 0:
                if not haszero:
                    so.insert(-1, num[0])
                    haszero = True
            else:
                so.append(num[n])
                haszero = False
        else:
            if n != 0:
                so.append(iunit[i])
                so.append(num[n])
                haszero = False
            else:
                if not haszero:
                    so.append(num[0])
                    haszero = True

    so.reverse()
    return ''.join(so)


def batch_get_chn_num_roi(img_dir, json_dir, save_dir):
    '''
    crcb shrd data 数字标注转中文大写标注
    '''
    img_name_list = os.listdir(img_dir)

    for item in img_name_list:
        tmp_label = item.split('_')
        pre_label = tmp_label[0] + '_' + tmp_label[1] + '_'
        label_value = tmp_label[2].split('.png')[0]
        img_path = os.path.join(img_dir, item)
        json_path = os.path.join(json_dir, item+'.json')

        save_roi_path = os.path.join(save_dir, pre_label + cncurrency(label_value) + '.png')

        get_chn_num(img_path, json_path, save_roi_path)

if __name__ == '__main__':
    date_key = '20180731'
    img_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/CA'
    item_list = ['CA', 'Ck', 'DR']
    # item_list = ['CA']

    for item in item_list:
        img_dir = os.path.join(img_root, item, date_key, 'mark')
        save_dir = os.path.join(img_root, item, date_key, 'roi')
        json_dir = os.path.join(img_root, item, date_key, 'json')

        logging.info("Deal with the folder: %s" %(img_dir))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        batch_get_chn_num_roi(img_dir, json_dir, save_dir)
    