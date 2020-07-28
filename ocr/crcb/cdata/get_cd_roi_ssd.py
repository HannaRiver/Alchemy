#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'detection'))
# os.environ['GLOG_minloglevel'] = '2'

import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

import cv2
from ssd_detect import CaffeDetection
from get_ca_roi_ssd import save_single_roi


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/tmp/SignatureRec/model/DateRecog/labelmap.prototxt')
    parser.add_argument('--mean_value', default=[185, 185, 185])
    parser.add_argument('--model_def',
                        default='/work/tmp/SignatureRec/model/DateRecog/DateRecogSSD.prototxt')
    parser.add_argument('--image_resize', default=[160, 480], type=int)
    parser.add_argument('--model_weights',
                        default='/work/tmp/SignatureRec/model/DateRecog/DateRecogSSD.caffemodel')
    return parser.parse_args()


def bacth_save_ssd_roi(data_root, item_list, date_list):
    args = parse_args()
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    
    # data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/CA'
    # item_list = ['CA', 'Ck', 'DR']
    # # date_list = ['20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']
    # date_list = ['20180731']
    
    for item in item_list:
        for date in date_list:
            data_dir = os.path.join(data_root, item, date)
            img_name_list = os.listdir(os.path.join(data_dir, 'mark'))
            save_path_dir = os.path.join(data_dir, 'ssd_roi')

            if not os.path.exists(save_path_dir):
                os.mkdir(save_path_dir)

            logging.info("Deal with the file: ./%s/%s[size: %s] \n" %(item, date, len(img_name_list)))

            for img_name in img_name_list:
                # img_name = img_name.decode('ascii').encode('utf-8')
                image_file = os.path.join(data_dir, 'mark', img_name)
                if not os.path.exists(image_file):
                    continue
                save_path = os.path.join(save_path_dir, img_name)

                save_single_roi(image_file, detection, save_path)
    
    logging.info("Batch get ssd result roi done!-> CA Ck DR")

def bacth_save_ssd_roi2():
    pass

if __name__ == '__main__':
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/Date'
    item_list = ['支票签发日期']
    date_list = ['20180731', '20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']
    bacth_save_ssd_roi(data_root, item_list, date_list)

