#/usr/bin/python3
#-*- coding=utf-8 -*-

import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..','..', '..', 'utils')))

import cv2
from file2list import readTxt

def diff_filename_4folder(folder, subset_folder):
    '''
    两个文件夹的差异，返回folder - subset_folder
    '''
    return list(set(os.listdir(folder)) - set(os.listdir(subset_folder)))

def main(v):
    pics_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch' + v + '/pics'
    img_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch' + v + '/roi_batch' + v
    sub_folder_dir = '/work/hena/ocr/data/CLPR/shuzhou_20180531/batch' + v + '/det_v002_error'

    model_det_txt_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch' + v + '/txt'

    save_txt_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch' + v + '/v002_txt'
    save_clpr_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch' + v + '/roi_model'

    if not os.path.exists(save_txt_dir):
        os.mkdir(save_txt_dir)
    if not os.path.exists(save_clpr_dir):
        os.mkdir(save_clpr_dir)

    right_list = diff_filename_4folder(pics_dir, sub_folder_dir)

    for item in right_list:
        imgname = item.split('.')[0]

        save_txt_path = os.path.join(save_txt_dir, imgname + '.txt')
        img_path = os.path.join(img_dir, item)

        # 保存模型定位正确的txt
        shutil.move(os.path.join(model_det_txt_dir, imgname + '.txt'), save_txt_path)

        # 保存roi
        x, y, w, h = [int(i) for i in readTxt(save_txt_path)[0].split(' ')]
        img = cv2.imread(img_path)
        roi_img = img[y: y + h, x: x + w]
        cv2.imwrite(os.path.join(save_clpr_dir, item), roi_img)
        

        
if __name__ == "__main__":
    main(sys.argv[1])


        
