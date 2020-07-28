#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import shutil
"""
功能: 整理车检已确认/修改标签后的数据，将车牌(roi)对应的车辆图片及定位txt保存在一个batch中
"""


def get_label4name(namestr, idx=0, split_key='_'):
    """
    通过解析文件名获取图像标签
    idx: 需要的内容在第几个项
    split_key: 分割字符串的key
    """
    if split_key not in namestr:
        return False
    return namestr.split(split_key)[idx]

def sorting_checked_vaidata(car_dir, plate_dir, rect_txt_dir):
    save_car_dir = os.path.join(plate_dir, '..', 'car')
    save_rect_dir = os.path.join(plate_dir, '..', 'plate_vairect_txt')
    if not os.path.exists(save_car_dir):
        os.mkdir(save_car_dir)
    if not os.path.exists(save_rect_dir):
        os.mkdir(save_rect_dir)
    
    for item in os.listdir(plate_dir):
        plate_num = get_label4name(item, 0, '_')
        if not plate_num:
            continue
        img_name = item[len(plate_num)+1: ]

        car_path = os.path.join(car_dir, img_name)
        rect_txt_path = os.path.join(rect_txt_dir, img_name)

        if not os.path.exists(car_path) or not os.path.exists(rect_txt_path):
            continue

        shutil.copy(car_path, os.path.join(save_car_dir, img_name))
        shutil.copy(rect_txt_path, os.path.join(save_rect_dir, img_name))

def main():
    plate_dir = '/work/hena/ocr/data/CLPR/VAI/batch2_20190121/batch0_20190122/plate'
    car_dir = '/work/hena/ocr/data/CLPR/VAI/batch2_20190121/car'
    rect_txt_dir = '/work/hena/ocr/data/CLPR/VAI/batch2_20190121/txt'

if __name__ == '__main__':
    main()