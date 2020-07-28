#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'detection'))
os.environ['GLOG_minloglevel'] = '2'

import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

import cv2
import shutil
from ssd_detect import CaffeDetection
from ssd_detect import get_bnbox
import random

def save_ssd_result(args, image_dir, save_dir):
    '''将ssd定位的结果画出来并保存'''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        save_path = os.path.join(save_dir, img_name)
        get_bnbox(img_path, detection, save_path)

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/tmp/SignatureRec/model/CapitalMoneyRecog/labelmap.prototxt')
    parser.add_argument('--mean_value', default=[185, 185, 185])
    parser.add_argument('--model_def',
                        default='/work/hena/ocr/model/caffe/HandWriting/locate/V0.0.1/CapitalMoneyRecogSSD.prototxt')
    parser.add_argument('--image_resize', default=[160, 480], type=int)
    parser.add_argument('--model_weights',
                        default='/work/hena/ocr/model/caffe/HandWriting/locate/V0.0.1/CapitalMoneyRecogSSD.caffemodel')
    return parser.parse_args()
    
def batch_get_aug_roi(image_file, detection, save_dir, date, expand_rate=0.05):
    result = detection.detect(image_file)
    if len(result) == 0:
        return False
    result = result[0]

    item = os.path.basename(image_file)

    img = cv2.imread(image_file)
    height, width = img.shape[: 2]
    xmin, xmax = [round(result[i] * width) for i in [0, 2]]
    ymin, ymax = [round(result[i] * height) for i in [1, 3]]
    w, h = xmax - xmin, ymax - ymin

    xmin = int(max(0, xmin - w*expand_rate))
    xmax = int(min(width, xmax + w*expand_rate))
    ymin = int(max(0, ymin - h*expand_rate))
    ymax = int(min(height, ymax + h*expand_rate))

    roi_x = xmin
    roi_y = ymin
    roi_w = xmax - xmin
    roi_h = ymax - ymin

    for i in range(10):
        random_x1 = int(10 * random.random() - 5)
        random_x2 = int(10 * random.random() - 5)
        random_y1 = int(10 * random.random() - 5)
        random_y2 = int(10 * random.random() - 5)

        roi_x = max(roi_x - random_x1, 0)
        roi_y = max(roi_y - random_y1, 0)
        roi_w = min(roi_w + random_x1 + random_x2, width-roi_x)
        roi_h = min(roi_h + random_y2 + random_y1, height-roi_y)
        key = date + '_' + str(random_x1) + str(random_x2) + str(random_y1) + str(random_y1) + '_'

        save_roi_path = os.path.join(save_dir, key + item)
        roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        cv2.imwrite(save_roi_path, roi_img)



def save_single_roi(image_file, detection, save_path='', expand_rate=0.05):
    result = detection.detect(image_file)
    if len(result) == 0:
        return False
    result = result[0]

    img = cv2.imread(image_file)
    height, width = img.shape[: 2]
    xmin, xmax = [round(result[i] * width) for i in [0, 2]]
    ymin, ymax = [round(result[i] * height) for i in [1, 3]]

    if xmin < 0 or ymin < 0 or ymax <= ymin or xmax <= xmin:
        return False

    w, h = xmax - xmin, ymax - ymin

    xmin = int(max(0, xmin - w*expand_rate))
    xmax = int(min(width, xmax + w*expand_rate))
    ymin = int(max(0, ymin - h*expand_rate))
    ymax = int(min(height, ymax + h*expand_rate))

    roi_img = img[ymin: ymax, xmin: xmax]

    cv2.imwrite(save_path, roi_img)

    return True

def test_function(data_root, item_list, date):
    args = parse_args()
    for item in item_list:
        image_dir = os.path.join(data_root, item, date+'check_result')
        save_dir = os.path.join(data_root, item, date+'check_bnbox')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_ssd_result(args, image_dir, save_dir)

def bacth_save_ssd_roi():
    args = parse_args()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/CA'
    save_dir = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/CA_V011_aug726102224'
    ssd_version = 'v001'
    item_list = ['CA', 'Ck', 'DR']
    # date_list = ['20180906', '20180907', '20180910', '20180911', '20180913', '20180914', '20180917', '20180918', '20180919', '20180920', '20180925',
    #               '20181009', '20181011', '20181015', '20181018', '20180711', '20180720', '20180724', '20180725', '20180727', '20180730']
    # date_list = ['20180731', '20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']
    date_list = ['20180726', '20181022', '20181024']
    for item in item_list:
        for date in date_list:
            data_dir = os.path.join(data_root, item, date)
            img_dir = os.path.join(data_dir, 'img')
            # data_dir = os.path.join(data_root, item, 'labeled', date)
            # img_dir = os.path.join(data_dir, 'mark')

            img_name_list = os.listdir(img_dir)
            save_path_dir = os.path.join(data_dir, ssd_version+'_ssd_roi')

            if not os.path.exists(save_path_dir):
                os.mkdir(save_path_dir)

            logging.info("Deal with the file: ./%s/%s[size: %s] \n" %(item, date, len(img_name_list)))

            for img_name in img_name_list:
                image_file = os.path.join(img_dir, img_name)
                if not os.path.exists(image_file):
                    logging.info("Warning: File not exists! -> %s" %(image_file))
                    continue
                save_path = os.path.join(save_path_dir, img_name)

                # save_single_roi(image_file, detection, save_path)
                batch_get_aug_roi(image_file, detection, save_dir, date)
    
    logging.info("Batch get ssd result roi done!-> CA Ck DR")

def bacth_save_ssd_roi2(data_root, item_list, date):
    args = parse_args()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)

    for item in item_list:
        data_dir = os.path.join(data_root, item, date)
        save_dir = os.path.join(data_root, item, date+'ssd_result')
        label_dir = os.path.join(data_root, item, date+'need_label')

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        
        img_name_list = os.listdir(data_dir)

        logging.info("Deal with the file: ./%s/%s[size: %s] \n" %(item, date, len(img_name_list)))

        for img_name in img_name_list:
            image_file = os.path.join(data_dir, img_name)
            if not os.path.exists(image_file):
                continue
            save_path = os.path.join(save_dir, img_name)

            flag = save_single_roi(image_file, detection, save_path)
            if flag == False:
                shutil.copy(image_file, os.path.join(label_dir, img_name))
                

if __name__ == '__main__':
    # data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item'
    # item_list = ['大写金额', '支票大写金额', '进账单大写金额']
    # date = '20181029'
    # bacth_save_ssd_roi2(data_root, item_list, date)
    # bacth_save_ssd_roi()
    # test_function(data_root, item_list, date)
    save_ssd_result(parse_args(), '/work/ocr/handwriting/data/bill/CRCB/crcb_shred/cls_item/付款人户名/20180720', '/work/ocr/handwriting/data/bill/CRCB/crcb_shred/cls_item/大写金额/test')