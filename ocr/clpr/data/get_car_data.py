#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))

from txt2roi import get_car_obj
from file2list import readTxt
import shutil


def batch_get_car_obj(image_dir, txt_dir, save_roi_dir):
    if os.path.isdir(txt_dir):
        txt_list = os.listdir(txt_dir)
    elif os.path.isfile(txt_dir) and txt_dir[-3: ] == 'txt':
        txt_list = readTxt(txt_dir)
        if '/' in txt_list[0]:
            txt_list = [i.split('/')[-1] for i in txt_list]
    else:
        print("不支持标签这种输入，请输入文件路径或列表.txt")
        return
    
    for item in txt_list:
        img_path = os.path.join(image_dir, item[: -4] + '.jpg')
        txt_path = os.path.join(txt_dir, item)
        save_roi_path = os.path.join(save_roi_dir, item[: -4] + '.jpg')

        get_car_obj(img_path, txt_path, save_roi_path)
        
def get_batch_data(root_dir, batch):
    file_list = readTxt(os.path.join(root_dir, 'batch' + str(batch) + 'k.txt'))
    img_dir = os.path.join(root_dir, 'image_batch' + str(batch))
    result_dir = os.path.join(root_dir, 'result_batch' + str(batch))
    txt_dir = os.path.join(root_dir, 'txt_batch' + str(batch))

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
        os.mkdir(result_dir)
        os.mkdir(txt_dir)
    
    
    for item in file_list:
        image_name = os.path.basename(item).split('.')[0]
        image_path = os.path.join(root_dir, 'image', image_name + '.jpg')
        result_path = os.path.join(root_dir, 'result', image_name + '.jpg')
        txt_path = os.path.join(root_dir, 'txt', item)
        if not os.path.exists(image_path) or not os.path.exists(result_path):
            print(item, image_name)
            continue
        shutil.move(image_path, os.path.join(img_dir, image_name + '.jpg'))
        shutil.move(result_path, os.path.join(result_dir, image_name + '.jpg'))
        shutil.move(txt_path, os.path.join(txt_dir, image_name + '.txt'))
    



def main(batch):
    # 要自己先新建一个batchXk.txt 再运行
    data_root = '/work/hena/ocr/pro/CLPR/generate_dir'
    get_batch_data(data_root, batch)
    save_roi_dir = os.path.join(data_root, 'roi_batch' + str(batch))
    if not os.path.exists(save_roi_dir):
        os.mkdir(save_roi_dir)
    
    image_dir = os.path.join(data_root, 'image_batch' + str(batch))
    txt_dir = os.path.join(data_root, 'txt_batch' + str(batch))

    batch_get_car_obj(image_dir, txt_dir, save_roi_dir)


if __name__ == '__main__':
    main(sys.argv[1])