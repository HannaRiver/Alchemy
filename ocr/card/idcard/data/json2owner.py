#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import sys
import json
import random
import cv2
import numpy as np


def undeform_resize(img, resize_h, resize_w, mean_value=[150, 139, 138]):
    '''
    不变形填背景resize
    '''
    new_img = cv2.merge([np.ones((resize_h, resize_w)) * i for i in mean_value])
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


max_label = 0
def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def idcard_json2owner_rect(json_path, get_key='xingming'):
    location_x, location_y = [], []
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        owners = data[get_key]
        for owner in owners:
            owner_name = owner[get_key]
            location_key = get_key + '_location'
            name_location = owner[location_key]
            for item in name_location:
                locaton_add_x = [int(item["location"][i][0]) for i in range(4)]
                locaton_add_y = [int(item["location"][i][1]) for i in range(4)]
                location_x += locaton_add_x
                location_y += locaton_add_y
    return min(location_x), min(location_y), max(location_x), max(location_y), owner_name

def driving_license_json2rect_add(json_path, get_key='zhuzhi'):
    info = []
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        owners = data[get_key]
        for owner in owners:
            location_x, location_y = [], []
            owner_name = owner[get_key]
            location_key = get_key + '_location'
            name_location = owner[location_key]
            for item in name_location:
                locaton_add_x = [int(item["location"][i][0]) for i in range(4)]
                locaton_add_y = [int(item["location"][i][1]) for i in range(4)]
                location_x += locaton_add_x
                location_y += locaton_add_y
            info.append([min(location_x), min(location_y), max(location_x), max(location_y), owner_name])
    return info
                  
def json2idcard_roi_add(image_path, json_path, save_dir, get_key='zhuzhi'):
    global max_label
    img = cv2.imread(image_path)
    if not os.path.exists(json_path):
        return
    img_name = os.path.basename(image_path)
    info = driving_license_json2rect_add(json_path, get_key)
    for i, item in enumerate(info):
        xmin, ymin, xmax, ymax, label = item
        xmin, ymin = [i - random.randint(0, 5) for i in [xmin, ymin]]
        xmax, ymax = [i + random.randint(0, 5) for i in [xmax, ymax]]
        locate = str(xmin) + '_' + str(ymin) + '_' + str(xmax) + '_' + str(ymax)
        roi_img = img[ymin: ymax, xmin: xmax]
        # resized_img = undeform_resize(roi_img, 20, 400, [127.5, 127.5, 127.5])
        save_path = os.path.join(save_dir, label + '_'+ str(i) + '_' + locate + '_' + img_name)
        if len(label) > max_label:
            max_label = len(label)
        cv2.imwrite(save_path, roi_img)

def json2idcard_roi(image_path, json_path, save_dir, get_key='xingming'):
    global max_label
    img = cv2.imread(image_path)
    if not os.path.exists(json_path):
        return
    xmin, ymin, xmax, ymax, label = idcard_json2owner_rect(json_path, get_key)
    if len(label) > max_label:
        max_label = len(label)
    xmin, ymin = [i - random.randint(0, 5) for i in [xmin, ymin]]
    xmax, ymax = [i + random.randint(0, 5) for i in [xmax, ymax]]
    locate = str(xmin) + '_' + str(ymin) + '_' + str(xmax) + '_' + str(ymax)
    roi_img = img[ymin: ymax, xmin: xmax]
    # resized_img = undeform_resize(roi_img, 20, 400, [127.5, 127.5, 127.5])
    img_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, label + '_' + locate + '_' + img_name)
    cv2.imwrite(save_path, roi_img)

#OCR_Type2Item = {'drving_license': ["zhuzhi"],
#                 'template': ['至', 'validPeriod', '有效期限', 'class', '准驾车型', 'dateOfFirstIssde', '初次领证日期', 'dateOfBirth', '出生日期', 'address', '住址', 'nationality', '国籍', 'sex', '性别', '证号', 'name', '姓名']}
# OCR_Type2Item = {'drving_license': ["chexing", "chucilingzhengriqi", "chushengriqi", "guoji", "xingbie", "xingming", "youxiaoqiixian1", "youxiaoqiixian2", "zhenghao", "zhuzhi"]}
OCR_Type2Item = {'drving_license': ["zhuzhi", "xingming"]}

def get_commonocr_fark_roi(image_txt, json_dir, save_dir, ocr_type='drving_license'):
    print("Deal with %s" %(image_txt))
    image_path_list = readTxt(image_txt)
    item_list = OCR_Type2Item[ocr_type]
    for item in item_list:
        print("Deal with [%s] data..." %(item))
        save_item_dir = os.path.join(save_dir, item)
        if not os.path.exists(save_item_dir):
            os.mkdir(save_item_dir)
        for image_path in image_path_list:
            image_name, image_suffix = os.path.splitext(os.path.basename(image_path))
            json_path = os.path.join(json_dir, image_name + '.json')
            if item in ["zhuzhi"]:  
                json2idcard_roi_add(image_path, json_path, save_item_dir)
            else:
                json2idcard_roi(image_path, json_path, save_item_dir, item)
            

def get_template_fark_roi(image_path, json_path, save_root):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    img = cv2.imread(image_path)
    img_name = os.path.basename(image_path)
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        owners = data['objects']
        for owner in owners:
            owner_name = owner['label']
            location_key = 'rect'
            save_dir = os.path.join(save_root, owner_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            x, y, w, h = owner[location_key]
            for i in range(10):
                xmin, ymin, xmax, ymax = int(x), int(y), int(x+w), int(y+h)
                xmin, ymin = [i - random.randint(0, 7) for i in [xmin, ymin]]
                xmax, ymax = [i + random.randint(0, 7) for i in [xmax, ymax]]
                roi_img = img[ymin: ymax, xmin: xmax]
                save_path = os.path.join(save_dir, owner_name + '_' + str(i) + '_' + img_name)
                cv2.imwrite(save_path, roi_img)

def batch_get_commonocr_fark_roi(batch_key='1', ocr_type = 'drving_license'):
    data_root = '/work/ocr/card/driving_license/data/fake/jiashizheng0327'
    batch_info = 'batch' + batch_key
    # 'jiashizheng01', 'jiashizheng02',
    for i in ['jiashizheng01', 'jiashizheng02', 'jiashizheng03', 'jiashizheng04']:
        image_txt = os.path.join(data_root, i, batch_info, batch_info + '.txt')
        json_dir = os.path.join(data_root, i, 'log')
        save_dir = os.path.join(data_root, i, batch_info)
        
        get_commonocr_fark_roi(image_txt, json_dir, save_dir, ocr_type)
        print(max_label)


if __name__ == '__main__':
    batch_get_commonocr_fark_roi()
    # get_template_fark_roi(image_txt, json_dir, save_dir)
    