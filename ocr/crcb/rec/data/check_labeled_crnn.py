#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
import argparse
import shutil
import cv2
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
sys.path.append(os.path.join(tool_root, 'ocr', 'crcb', 'ca'))
from file2list import readTxt, get_label4name
from shred_recognise import get_recognition, refuse2plate
from resize_img import undeform_center_resize
from resize_img import map4undeform_center_resize
from lca2ca import cncurrency
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


# def check_label_args():
#     '''
#     分配数据基本的参数
#     '''
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_dir', default='/data1/ocr/data/org_print/checkdata', help='需要处理数据的地址')
#     parser.add_argument('--label_indx', type=int, default=0, help='碎片数据gt的位置')
#     parser.add_argument('--key_date', default='20181024', type=str, help='需要处理的日期批次号')
#     parser.add_argument('--version', default='v0.0.0', type=str, help='识别模型的版本号')
#     parser.add_argument('--rec_item', default='CA', help='处理的碎片大类[CA, LA, CD, LD, Num, Chn, Other]', type=str)
#     parser.add_argument('--debug_rec', default=False, help='是否看识别的具体结果', type=bool)
#     parser.add_argument('--showkey', default=False, help='是否看图', type=bool)

#     return parser.parse_args()

def shred_rec_args():
    '''
    常熟碎片识别模型参数(大写金额类-打印体)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/data1/ocr/data/org_print/checkdata', help='需要处理数据的地址')
    parser.add_argument('--label_indx', type=int, default=0, help='碎片数据gt的位置')
    parser.add_argument('--key_date', default='20181024', type=str, help='需要处理的日期批次号')
    parser.add_argument('--version', default='v0.0.0', type=str, help='识别模型的版本号')
    parser.add_argument('--rec_item', default='CA', help='处理的碎片大类[CA, LA, CD, LD, Num, Chn, Other]', type=str)
    parser.add_argument('--debug_rec', default=False, help='是否看识别的具体结果', type=bool)
    parser.add_argument('--showkey', default=False, help='是否看图', type=bool)

    parser.add_argument('--img_path', default='')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def',
                        default='/data1/ocr/net/CRNN/CA/V0.0.1/model/ca_v001_deploy.prototxt')
    parser.add_argument('--image_resize', default=[48, 240], type=int)
    parser.add_argument('--time_step', default=60)
    parser.add_argument('--scale', default=1)
    parser.add_argument('--resize_type', default='undeform_resize')
    parser.add_argument('--mean_value', default=[150, 139, 138])
    parser.add_argument('--model_weights',
                        default='/data1/ocr/net/CRNN/CA/V0.0.1/weights/ca_iter_140000.caffemodel')
    return parser.parse_args()

def normShredLabel(label, rec_item):
    '''
    对银行标签进行归一化转换
    '''
    if rec_item == 'CA':
        return cncurrency(label)
    elif rec_item == 'CD':
        pass
    else:
        pass
    return label

def CheckLabelRec(rec_item, recognition, img_dir, save_right_dir, save_error_dir, label_indx=0, _IS_DEBUG_REC_=False, showkey=False, version='V0.0.0'):
    if not os.path.exists(save_right_dir):
        os.mkdir(save_right_dir)
    if not os.path.exists(save_error_dir):
        os.mkdir(save_error_dir)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        label = get_label4name(os.path.splitext(img_name)[0], label_indx, '_')
        img = cv2.imread(img_path)
        pre_result, _, _, = refuse2plate(img, recognition, _IS_DEBUG_REC_)
        norm_label = normShredLabel(label, rec_item)
        save_name = norm_label + '_' + pre_result + '_' + img_name
        save_path = os.path.join(save_right_dir, save_name) if norm_label == pre_result else os.path.join(save_error_dir, save_name)
        shutil.copy(img_path, save_path)




def BatchCheckLabelRec(rec_item, img_root, save_right_root, save_error_root, label_indx=0, _IS_DEBUG_REC_=False, showkey=False, version='V0.0.0'):
    if not os.path.exists(save_right_root):
        os.mkdir(save_right_root)
    if not os.path.exists(save_error_root):
        os.mkdir(save_error_root)
    rec_args = shred_rec_args()
    recognition = get_recognition(rec_args) # 获取lstm识别器
    for folder in os.listdir(img_root):
        logging.info("-----> Deal with the folder %s" %(folder))
        img_dir = os.path.join(img_root, folder)
        save_right_dir = os.path.join(save_right_root, folder)
        save_error_dir = os.path.join(save_error_root, folder)
        CheckLabelRec(rec_item, recognition, img_dir, save_right_dir, save_error_dir, label_indx, _IS_DEBUG_REC_, showkey, version)

def main():
    args = shred_rec_args()
    rec_args = shred_rec_args()
    img_root = os.path.join(args.img_dir, args.rec_item, 'return', args.key_date)
    save_right_root = os.path.join(args.img_dir, args.rec_item, 'check_right', args.key_date)
    save_error_root = os.path.join(args.img_dir, args.rec_item, 'check_error', args.key_date)
    logging.info("==================== Check Labeled Rec Data By Model =====================")
    logging.info("Model Version: %s \t Model Name: %s" %(args.version, os.path.basename(rec_args.model_weights)))
    logging.info("Mean Value: %s \t Scale: %s \t Resize: %s \t Resize Type: %s"  %(rec_args.mean_value, rec_args.scale, str(rec_args.image_resize), rec_args.resize_type))
    logging.info("处理的文件目录为: %s" %(img_root))
    logging.info("Model Right Root: %s" %(save_right_root))
    logging.info("Model Error Root: %s" %(save_error_root))
    BatchCheckLabelRec(args.rec_item, img_root, save_right_root, save_error_root, args.label_indx, args.debug_rec, args.showkey, args.version)
    logging.info("==================== Check Labeled Rec Data By Model Done =====================")

if __name__ == '__main__':
    main()