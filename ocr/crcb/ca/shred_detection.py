#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'caffe', 'detection'))
sys.path.append(os.path.join(tool_root, 'utils'))
os.environ['GLOG_minloglevel'] = '2'
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
import cv2
import shutil
from file2list import readTxt
from ssd_detect import CaffeDetection
from resize_img import undeform_center_resize
from resize_img import map4undeform_center_resize


def get_chars_bnbox(image_file, detection, args, save_path='', showkey=False):
    '''
    获取定位结果，获取每一个字符的bnbox
    return -> 按照x坐标顺序的bnbox list
    return -> [[xmin, xmax, ymin, ymax, label], ...] 对应原图坐标
    '''
    if showkey:
        logging.info("Deal %s ..." %(image_file))
    results = detection.detect(image_file, 0.2, 100)
    chars_bnboxs = []
    if len(results) == 0:
        if showkey:
            logging.info("模型认为该图不存在目标 --> %s" %(image_file))
        if len(save_path) != 0 and os.path.exists(os.path.dirname(save_path)):
            # shutil.copy(image_file, save_path)
            pass
        return chars_bnboxs
    
    # 根据xmin对resut进行排序
    results.sort(key=lambda x: x[0])
    
    img = cv2.imread(image_file) if type(image_file) == type('Hi') else image_file
    for j, result in enumerate(results):
        show_img = img.copy()
        label = result[4]
        height, width = img.shape[: 2]

        xmin, xmax = [int(round(result[i] * width)) for i in [0, 2]]
        ymin, ymax = [int(round(result[i] * height)) for i in [1, 3]]

        if xmin < 0 or ymin < 0 or ymax <= ymin or xmax <= xmin or ymax-ymin < 2 or xmax-xmin < 2:
            continue
        
        chars_bnboxs.append([xmin, xmax, ymin, ymax, label])
        color = (0, 0, 255) if label == 1 else (255, 0, 0)
        if showkey:
            color = (0, 0, 255) if label == 1 else (255, 0, 0)
            cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), color, 1)

            cv2.namedWindow("char segment result", cv2.WINDOW_NORMAL)
            cv2.imshow("char segment result", show_img)
            cv2.waitKey(0)
            logging.info("[%s th] socre: %s \t label: %s" %(str(j), str(result[5]), str(label)))
        
        if len(save_path) != 0:
            if False:
                roi = img[ymin: ymax, xmin: xmax]
                save_dir, save_name = os.path.dirname(save_path), os.path.basename(save_path)
                cv2.imwrite(os.path.join(save_dir, str(j) + '_' + save_name), roi)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
    
    if len(save_path) != 0 and os.path.exists(os.path.dirname(save_path)):
        pass
        cv2.imwrite(save_path, img)
    
    return chars_bnboxs

def get_shred_bnboxs(args, test_root_dir='', _IS_SAVE_=False, showkey=False, version='v000'):
    test_root_dir = args.img_dir if len(test_root_dir) == 0 else test_root_dir
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file, args.mean_value, args.input_scale, args.resize_type)
    save_dir = os.path.join(test_root_dir, '..', version) if test_root_dir[-3: ] != 'txt' else os.path.join(os.path.dirname(test_root_dir), version)
    if not os.path.exists(save_dir) and _IS_SAVE_:
        os.mkdir(save_dir)
    
    if os.path.isfile(test_root_dir):
        if test_root_dir[-3: ] != 'txt':
            img_path_list = [test_root_dir]
        else:
            img_path_list = readTxt(test_root_dir)
        IsFile = True
    elif os.path.isdir(test_root_dir):
        img_path_list = os.listdir(test_root_dir)
        IsFile = False
    else:
        pass
        print("Error: 不支持的路径输入 -> %s" %(test_root_dir))
    
    chars_bnbox_list = []
    img_abs_paths = []
    # img_path_list = os.listdir(test_root_dir)
    for item in img_path_list:
        image_file = os.path.join(test_root_dir, item) if not IsFile else item
        img_abs_paths.append(image_file)
        save_path = os.path.join(save_dir, item) if _IS_SAVE_ else ''
        chars_bnbox = get_chars_bnbox(image_file, detection, args, save_path, showkey)
        chars_bnbox_list.append(chars_bnbox)
    return chars_bnbox_list, img_abs_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--img_dir', type=str, default='/work/ocr/card/driving_license/data/seg/test', help='需要分割的图片根目录')
    parser.add_argument('--labelmap_file',
                        default='/work/ocr/card/model/chars_segment/V0.0.1/model/chars_seg_v001_labelmap_voc.prototxt')
    parser.add_argument('--mean_value', default=[145, 140, 144])
    parser.add_argument('--input_scale', default=1)
    parser.add_argument('--model_def',
                        default='/work/ocr/card/model/chars_segment/V0.0.1/model/chars_seg_v001_deploy.prototxt')
    parser.add_argument('--image_resize', default=[300, 300], type=int)
    parser.add_argument('--resize_type', default='undeform_center', type=str)
    parser.add_argument('--model_weights',
                        default='/work/ocr/card/model/chars_segment/V0.0.1/model/chars_seg_v001_iter_140000.caffemodel')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    for i in ['1PlateNo', '9RegisterDate', '10IssueDate', '4address', '5UseCharacter', '8EngineNo', '7VIN', '3Owner', '2VehicleType', '6Model']:
        logging.info("Deal with the folder: %s" %(i))
        chars_bnbox_list, img_path_list = get_chars_bnboxs(args, '/work/ocr/card/vehicle_license/data/test/cut/' + i + '/ok', _IS_SAVE_=True, showkey=False, version='char_seg_v001_chars_1')
    logging.info("Chars segment work done!")