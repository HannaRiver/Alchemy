#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import sys
import cv2
import argparse
import logging
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))

from json2roi import json2allroi

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def det_labeld_args():
    '''
    标注了定位结果后分发的参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/data1/ocr/data/det', help='需要处理数据的地址')
    parser.add_argument('--save_dir', default='/data1/ocr/data/org_print/checkdata', help='需要处理数据的保存地址')
    parser.add_argument('--label_indx', type=int, default=-1, help='碎片数据gt的位置')
    parser.add_argument('--key_date', default='20181024', type=str, help='需要处理的日期批次号')
    parser.add_argument('--rec_item', default='CA', help='处理的碎片大类[CA, LA, CD, LD, Num, Chn, Other]', type=str)
    return parser.parse_args()

def GetRoi4Json(img_dir, json_dir, save_dir, label_indx=-1):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for img_name in os.listdir(img_dir):
        label = os.path.splitext(img_name)[0].split('_')[label_indx]
        img_path = os.path.join(img_dir, img_name)
        json_path = os.path.join(json_dir, img_name + '.json')
        if not os.path.exists(json_path):
            logging.info("Warring:: %s json file not exists！" %(json_path))
            continue
        rois = json2allroi(json_path)
        img = cv2.imread(img_path)
        for roi in rois:
            roi_x, roi_y, roi_w, roi_h = roi
            roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
            roi_name = '_'.join([label] + [str(i) for i in roi] + [img_name])
            save_path = os.path.join(save_dir, roi_name)
            cv2.imwrite(save_path, roi_img)

def BatchGetRoi4Json(img_root, save_root, label_indx=-1):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for item in os.listdir(img_root):
        logging.info("-----> Deal with the folder %s" %(item))
        img_dir = os.path.join(img_root, item, 'mark')
        json_dir = os.path.join(img_root, item, 'json')
        save_dir = os.path.join(save_root, item)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        GetRoi4Json(img_dir, json_dir, save_dir, label_indx)


def main():
    args = det_labeld_args()
    img_root = os.path.join(args.img_dir, args.rec_item, args.key_date, 'result')
    save_root = os.path.join(args.save_dir, args.rec_item, 'return', args.key_date)
    logging.info("==================== Get Roi From Json =====================")
    logging.info("定位标注完毕地址为: %s \t 识别项: %s \t 日期: %s" %(args.img_dir, args.rec_item, args.key_date))
    logging.info("处理后的图片保存路径: %s" %(save_root))
    BatchGetRoi4Json(img_root, save_root, args.label_indx)
    logging.info("==================== Get Roi From Json  Done =====================")

if __name__ == '__main__':
    main()