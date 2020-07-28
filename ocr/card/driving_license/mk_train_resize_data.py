#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import sys
import math
import cv2
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from resize_img import undeform_resize
from file2list import readTxt


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
        image_file = os.path.join(img_dir, item)
        if IsFile:
            image_file = item
        img_name = os.path.basename(image_file)
        img = cv2.imread(image_file)
        if type(img) == type(cv2.imread('hi')):
            os.remove(image_file)
            continue
        resized_img = undeform_resize(img, 20, 400, [127.5, 127.5, 127.5])
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, resized_img)

def bath_mk_trian_resize_data(key_id=1):
    save_dir = '/work/ocr/card/driving_license/data/fake/jiashizheng01/jiashizheng0' + str(key_id) + '/train_data'
    img_root = '/work/ocr/card/driving_license/data/fake/jiashizheng01/jiashizheng0' + str(key_id) + '/batch2'
    for item in os.listdir(img_root):
        img_dir = os.path.join(img_root, item)
        mk_trian_resize_data(img_dir, save_dir)

def main():
    bath_mk_trian_resize_data(1)
    bath_mk_trian_resize_data(2)
    bath_mk_trian_resize_data(3)
    bath_mk_trian_resize_data(4)

if __name__ == '__main__':
    main()
        