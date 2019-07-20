#/usr/bin/python3
#-*- coding=utf-8 -*-
'''
对扣取的数据生成网络学习所需要的数据且做位置绕动拓增
且每次会按照1：4的比率分test：train数据，且拓增不相交
test部分不拓增
'''
import os
import shutil
import sys
import argparse
import cv2
import random
import numpy as np
import math
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
from alchemy_config import cfg
sys.path.append(cfg.UTILS_DIR)
sys.path.append(os.path.join(cfg.IMGAUG_DIR, 'rot'))
from getQuadRoi import get_quad_roi
from resize_img import undeform_resize


RecItem2Cfg = {
    'CA': cfg.CRCB_CFG_NAME.CA_PRINT,
    'LA': cfg.CRCB_CFG_NAME.LA_PRINT,
    'CD': cfg.CRCB_CFG_NAME.CD_PRINT,
    'LD': cfg.CRCB_CFG_NAME.LD_PRINT,
    'Chn': cfg.CRCB_CFG_NAME.CHN_PRINT,
    'Num': cfg.CRCB_CFG_NAME.NUM_PRINT,
}

def augResize_args():
    parser = argparse.ArgumentParser()
    # 处理原始数据目录所需要的参数
    parser.add_argument('--check_data_dir', default=cfg.CRCB_DATA.CHECKDATA_DIR_PRINT, help='所有需要核实数据的根目录')
    parser.add_argument('--item_type', default=cfg.CRCB_DATA.RECITEM, help='需要处理的大类，目前支持[CA:大写金额类, LA:小写金额类, LD: 小写日期, CD: 大写日期, Num: 数字类, Chn: 中文类, Other: 其他类]')
    parser.add_argument('--check_state', default=cfg.CRCB_DATA.CHECKSTATE, help='是否人工确认标签无误, 支持[check_right, check_error, model_right]')
    parser.add_argument('--date', default=cfg.CRCB_DATA.KEYDATE, help='需要处理的日期key')

    # 拓增所需要的参数
    parser.add_argument('--download_dir', default=cfg.CRCB_DATA.DOWNLOAD_DIR_PRINT, help='原始完整数据所在的根目录')
    parser.add_argument('--image_resize', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_RESIZESIZE, type=int)
    parser.add_argument('--mean_value', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_MEANVALUE)
    parser.add_argument('--resize_type', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_RESIZETYPE)

    # 保存处理后数据所需要的参数
    parser.add_argument('--save_dir', default=cfg.CRCB_DATA.TRIAN_REC_DIR_PRINT, help='处理数据保存地址')
    parser.add_argument('--cfg_path', default='')
    
    return parser.parse_args()

def AugResize2Train(check_state_dir, image_resize, mean_value, save_train_dir, save_test_dir, download_dir, save_bak_dir='', ratio=0.2, aug_times=5):
    img_list = os.listdir(check_state_dir)
    logging.info("========== Deal with the folder[%s](size: %s) ==========" %(os.path.basename(check_state_dir), len(img_list)))
    if not img_list:
        return
    for i, img_name in enumerate(img_list):
        # 直接resize现有定位后数据
        img_path = os.path.join(check_state_dir, img_name)
        img = cv2.imread(img_path)
        resized_img = undeform_resize(img, image_resize[0], image_resize[1], mean_value)
        save_path = os.path.join(save_test_dir, img_name) if i < len(img_list)*ratio else os.path.join(save_train_dir, img_name)
        cv2.imwrite(save_path, resized_img)

        if i < len(img_list)*ratio:
            continue
        # ====== Trian Part Aug =====
        # 伍万元整_伍万元整_50000.00_198_41_118_30_52_大写金额_50000.00.png
        img_name_infos =img_name.split('_') if '_' in img_name else [img_name] # 名字中所蕴含的信息
        label = img_name_infos[0] # 伍万元整
        org_img_name = '_'.join(img_name_infos[-3: ]) # 原始图像所对应的图片名 52_大写金额_50000.00.png
        roi_x_o, roi_y_o, roi_w_o, roi_h_o = [int(i) for i in img_name_infos[-7: -3]]
        
        # 找原图
        org_img_path = os.path.join(download_dir, org_img_name)
        if not os.path.exists(org_img_path):
            logging.warning("ori data %s not exists" %(org_img_path))
            continue
        org_img = cv2.imread(org_img_path)
        img_h, img_w = org_img.shape[: 2]
        # 旋转和水平拓增
        for j in range(aug_times):
            x1 = max(int(roi_x_o) - random.randint(0, 5), 0)
            x4 = max(int(roi_x_o) - random.randint(0, 5), 0)
            y1 = max(int(roi_y_o) - random.randint(0, 5), 0)
            y2 = max(int(roi_y_o) - random.randint(0, 5), 0)
            x2 = int(roi_x_o) + int(roi_w_o) + random.randint(0, 5)
            x3 = int(roi_x_o) + int(roi_w_o) + random.randint(0, 5)
            y3 = int(roi_y_o) + int(roi_h_o) + random.randint(0, 5)
            y4 = int(roi_y_o) + int(roi_h_o) + random.randint(0, 5)
    
            quad = [x1, y1, x2, y2, x3, y3, x4, y4]
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            xmax = max(x1, x2, x3, x4)
            ymax = max(y1, y2, y3, y4)
            if xmin < 1 or ymin < 1 or xmax > img_w or ymax > img_h:
                logging.warning("rect error")
                continue
            roi = org_img[ymin: ymax, xmin: xmax]
            new_roi = get_quad_roi(quad, org_img)
            locate = '_'.join([str(i) for i in quad])

            save_path = os.path.join(save_train_dir, label + '_' + locate + '_' + org_img_name)
            save_bak_path = os.path.join(save_bak_dir, label + '_' + locate + '_' + org_img_name)
            cv2.imwrite(save_bak_path, new_roi)
            new_aug_resized_img = undeform_resize(new_roi, image_resize[0], image_resize[1], mean_value)
            cv2.imwrite(save_path, new_aug_resized_img)

            new_resized_img = undeform_resize(roi, image_resize[0], image_resize[1], mean_value)
            save_roi_path = os.path.join(save_train_dir, label + '_rect_' + locate + '_' + org_img_name)
            save_bak_roi_path = os.path.join(save_bak_dir, label + '_rect_' + locate + '_' + org_img_name)
            cv2.imwrite(save_bak_roi_path, roi)
            cv2.imwrite(save_roi_path, new_resized_img)

def BatchAugResize2Train(check_state_root, image_resize, mean_value, save_train_dir, save_test_dir, download_root, save_bak_dir):
    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)
    if not os.path.exists(save_bak_dir):
        os.makedirs(save_bak_dir)
    for item in os.listdir(check_state_root):
        check_state_dir = os.path.join(check_state_root, item)
        download_dir = os.path.join(download_root, item)
        AugResize2Train(check_state_dir, image_resize, mean_value, save_train_dir, save_test_dir, download_dir, save_bak_dir)

def main():
    args = augResize_args()
    dates = [args.date] if ',' not in args.date else args.date.split(',') if type(args.date) == type('Hi') else args.date
    rec_item = args.item_type
    cfg_file = RecItem2Cfg[rec_item]
    for date in dates:
        check_state_root = os.path.join(args.check_data_dir, args.item_type, args.check_state, date)
        save_train_dir = os.path.join(args.save_dir, args.item_type, 'train', date, args.check_state)
        save_bak_dir = os.path.join(args.save_dir, args.item_type, 'train_need2aug', date)
        save_test_dir = os.path.join(args.save_dir, args.item_type, 'test', date)
        download_root = os.path.join(args.download_dir, args.item_type, date)
        BatchAugResize2Train(check_state_root, args.image_resize, args.mean_value, save_train_dir, save_test_dir, download_root, save_bak_dir)

if __name__ == '__main__':
    main()