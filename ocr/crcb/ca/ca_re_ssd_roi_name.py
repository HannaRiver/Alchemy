#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import sys
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def ca_re_ssd_roi_name(data_root, item_list, date_list):
    for item in item_list:
        for date in date_list:
            data_dir = os.path.join(data_root, item, date)
            img_name_dir = os.path.join(data_dir, 'ssd_roi')
            label_path = os.path.join(data_dir, date + '_roi.txt')

            img_name_list = os.listdir(img_name_dir)
            label_list = readTxt(label_path)

            logging.info("Deal with the file: ./%s/%s[size: %s] \n" %(item, date, len(img_name_list)))

            for img_item in img_name_list:
                img_index = int(img_item[: -4])
                re_img_name = label_list[img_index]

                os.rename(os.path.join(img_name_dir, img_item), os.path.join(img_name_dir, re_img_name))

if __name__ == '__main__':
    # data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/CA'
    # item_list = ['CA', 'Ck', 'DR']
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/Date'
    item_list = ['凭证日期', '支票签发日期']
    date_list = ['20180731', '20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']
    ca_re_ssd_roi_name(data_root, item_list, date_list)