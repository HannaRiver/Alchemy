#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import glob
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def get_label4name(namestr, idx=0, split_key='_'):
    """
    通过解析文件名获取图像标签
    idx: 需要的内容在第几个项
    split_key: 分割字符串的key
    """
    if split_key not in namestr:
        return False
    return namestr.split(split_key)[idx]

def folder_double_plate_rename(named_dir, ori_data_dir, pic_data_dir):
    """
    文件夹为单位改车检双黄车牌名字
    """
    for item in os.listdir(named_dir):
        named_img_path = os.path.join(named_dir, item)
        label = get_label4name(item, 0, '_')
        if not label:
            logging.info("Warrning: %s not be named!" %(named_img_path))
            continue
        ori_name = item[len(label)+1: ]

        # 原始的图片地址
        ori_img_path = os.path.join(ori_data_dir, ori_name)
        if not os.path.exists(ori_img_path):
            logging.info("Warrning: ori img(%s) not exists!" %(ori_img_path))
            continue

        pic_data_dir2 = pic_data_dir + '_error_checked'
        pic_data_path_list = glob.glob(pic_data_dir + "/*" + ori_name) + glob.glob(pic_data_dir2 + "/*" + ori_name)
        if len(pic_data_path_list) == 0:
            os.rename(ori_img_path, os.path.join(ori_data_dir, item))
        else:
            label = get_label4name(os.path.basename(pic_data_path_list[0]), 0, '_')
            if not label:
                logging.info("Warrning: %s not be named!" %(pic_data_path_list[0]))
                os.rename(ori_img_path, os.path.join(ori_data_dir, item))
            else:
                os.rename(ori_img_path, os.path.join(ori_data_dir, label + '_' + ori_name))

        
        

def main(named_root, ori_data_root, pic_data_root):
    # 根据蒋章的结果对数据进行第一次重命名 并确认是否修改了名字
    data_item_list = os.listdir(named_root)
    logging.info("===== Deal with VAI Double_yellow plate Data ... =====")
    logging.info("data root: %s\t size: %s" %(named_root, len(data_item_list)))
    for i, item in enumerate(data_item_list):
        named_dir = os.path.join(named_root, item)
        logging.info("[%s]Deal with folder(%s)..." %(i, item))
        ori_data_dir = os.path.join(ori_data_root, item, 'det_roi')
        pic_data_dir = os.path.join(pic_data_root, item.split('_')[0])
        folder_double_plate_rename(named_dir, ori_data_dir, pic_data_dir)




if __name__ == '__main__':
    # 被蒋章重命名的数据根目录
    named_root = '/work/hena/ocr/data/CLPR/VAI/double_yellow/chepai_tied_ssd'
    # 原始车检双黄车牌数据根目录
    ori_data_root = '/work/hena/ocr/data/CLPR/VAI/double_yellow/chepai'
    # 被何娜挑选出来并可能修改了标签的数据根目录
    pic_data_root = '/work/hena/ocr/data/CLPR/VAI/double_yellow'

    main(named_root, ori_data_root, pic_data_root)