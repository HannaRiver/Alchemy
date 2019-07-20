#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import sys
import math
import cv2
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from resize_img import undeform_resize
from file2list import readTxt


def getInfo(txtpath):
    result = {}
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            infos = line.strip().split(' ')
            if len(infos) != 2:
                print(line + " not ok!")
                continue
            img_name = infos[0]
            locate = infos[1]
            result[img_name] = locate
    return result

Dict_imgname_label = getInfo('/work/competitions/TinyMind/train_id_label_new.txt')


def mk_trian_resize_data(img_dir, save_dir):
    IsFile = False
    if os.path.isfile(img_dir):
        img_list = readTxt(img_dir)
        IsFile = True
    elif os.path.isdir(img_dir):
        img_list = os.listdir(img_dir)
    else:
        print("Error: 不支持的路径输入 -> %s" %(img_dir))

    print("Deal with the file: %s[size: %s]" %(os.path.basename(img_dir), len(img_list)))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("Resize后图片将会保存，保存地址为: %s\n" %(save_dir))

    for item in img_list:
        label = Dict_imgname_label[item] if item in Dict_imgname_label else ''
        if label == '':
            pass
            # continue
        image_file = os.path.join(img_dir, item)
        if IsFile:
            image_file = item
        img_name = os.path.basename(image_file)
        img = cv2.imread(image_file)
        if type(img) == type(cv2.imread('hi')):
            os.remove(image_file)
            continue
        resized_img = undeform_resize(img, 32, 160, [120, 127, 130])
        save_path = os.path.join(save_dir, label + '' +img_name)
        cv2.imwrite(save_path, resized_img)

def bath_mk_trian_resize_data(key_id=1):
    save_dir = '/work/competitions/TinyMind/data/rec/' + str(key_id) + '/ssd'
    img_root = '/work/competitions/TinyMind/PreTrainData/SSD'
    for item in os.listdir(img_root):
        img_dir = os.path.join(img_root, item)
        mk_trian_resize_data(img_dir, save_dir)

def main(key='5.0_8k'):
    img_dir, save_dir = '/work/competitions/TinyMind/result/CLPRSSD_' + key, '/work/competitions/TinyMind/data/rec/test/' + key
    mk_trian_resize_data(img_dir, save_dir)

if __name__ == '__main__':
    # ['0.1', '0.2', '0.5', '1.0', '2.0', '5.0', '10.0', '50.0', '100.0']
    # for i in ['10.0', '50.0', '100.0']:
    #     key = i + '_8k'
    #     main(key)
    # main('0.1_8k')
    img_dir = '/work/competitions/TinyMind/data/finals/CLPRSSD_private_test_data_8k'
    save_dir = '/work/competitions/TinyMind/data/finals/resized'
    mk_trian_resize_data(img_dir, save_dir)