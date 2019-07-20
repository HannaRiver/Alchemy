#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
import argparse
import cv2
from alchemy_config import cfg
utils_dir = cfg.UTILS_DIR
crcb_path = os.path.join(cfg.ROOT_DIR, 'ocr', 'crcb')
sys.path.append(utils_dir)
sys.path.append(crcb_path)
from file2list import readTxt, get_label4name
from rec.shred_recognise import get_recognition, refuse2plate
from det.shred_detection import get_shred_bnboxs
from utils.norm_shred_label import normShredLabel
from resize_img import undeform_center_resize
from resize_img import map4undeform_center_resize


def shred_det_args():
    '''
    常熟碎片定位模型参数(大写金额类-打印体)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_DET_LABELMAP)
    parser.add_argument('--mean_value', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_DET_MEANVALUE)
    parser.add_argument('--input_scale', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_DET_SCALE)
    parser.add_argument('--resize_type', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_DET_RESIZETYPE)
    parser.add_argument('--model_def', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_DET_MODELDEF)
    parser.add_argument('--image_resize', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_DET_RESIZESIZE, type=int)
    parser.add_argument('--model_weights', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_DET_MODELWEIGHT)
    return parser.parse_args()

def shred_rec_args():
    '''
    常熟碎片识别模型参数(大写金额类-打印体)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_MODELDEF)
    parser.add_argument('--image_resize', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_RESIZESIZE, type=int)
    parser.add_argument('--time_step', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_TIMESTEP)
    parser.add_argument('--scale', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_SCALE)
    parser.add_argument('--resize_type', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_RESIZETYPE)
    parser.add_argument('--mean_value', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_MEANVALUE)
    parser.add_argument('--model_weights', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_MODELWEIGHT)
    return parser.parse_args()

def check_label_args():
    '''
    分配数据基本的参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default=cfg.CRCB_DATA.DOWNLOAD_DIR_PRINT, help='需要处理数据的地址')
    parser.add_argument('--save_dir', default=cfg.CRCB_DATA.CHECKDATA_DIR_PRINT, help='需要处理数据的地址')
    parser.add_argument('--label_indx', type=int, default=-1, help='碎片数据gt的位置')
    parser.add_argument('--key_date', default=cfg.CRCB_DATA.DET2REC_KEYDATE, help='需要处理的日期批次号')
    parser.add_argument('--rec_item', default=cfg.CRCB_DATA.DET2REC_RECITEM, help='处理的碎片大类[CA, LA, CD, LD, Num, Chn, Other]', type=str)
    return parser.parse_args()

def det2Rec4Net(img_dir, _IS_SAVE_DET_=False, _IS_DEBUG_CLS_=False, _IS_DEBUG_REC_=False, showkey=False, version='char_seg_v000'):
    '''
    模型从定位到识别结果组合
    img_dir: 图片根目录，也支持图片list.txt, 也支持单张图片
    return : {'imgpath': [[xmin, xmax, ymin, ymax, label, rec], ...], }
    '''
    rec_args = shred_rec_args()
    det_args = shred_det_args()
    recognition = get_recognition(rec_args) # 获取lstm识别器
    # shred_bnbox_list = [[[xmin, xmax, ymin, ymax, label], ...], ...]
    # 得到定位结果
    shred_bnbox_list, img_path_list = get_shred_bnboxs(det_args, img_dir, _IS_SAVE_DET_, showkey, version)
    print("=====>>>>> Deal with folder/txt file: %s[size: %s]" %(img_dir, len(img_path_list)))

    # 下面是识别模块
    results = {} # {'imgpath': [[xmin, xmax, ymin, ymax, label, rec], ...], }
    for i, item in enumerate(shred_bnbox_list):
        infos = []
        img_path = img_path_list[i]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        for obj in item:
            info = obj
            xmin, xmax, ymin, ymax = obj[: 4]
            chars_img = img[ymin: ymax, xmin: xmax]
            # refuse2plate(image_file, lstm_classification, showkey=False)
            pre_result, _, _, = refuse2plate(chars_img, recognition, _IS_DEBUG_REC_)
            info.append(pre_result)
            infos.append(info)
        results[img_name] = infos
    return results

def det2Rec4LabelData(img_dir, save_right_dir, save_error_dir, rec_item='CA', label_indx=-1, version='v0.0.0'):
    # net_result = {'imgname': [[xmin, xmax, ymin, ymax, label, rec], ...], }
    net_result = det2Rec4Net(img_dir, False, False, False, False, version)

    for img_name in net_result:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        infos = net_result[img_name] # [[xmin, xmax, ymin, ymax, label, rec], ...]
        for item in infos:
            xmin, xmax, ymin, ymax, label, rec = item # [xmin, xmax, ymin, ymax, label, rec]
            label = get_label4name(os.path.splitext(img_name)[0], label_indx, '_')
            norm_label = normShredLabel(label, rec_item, rec)
            save_name = '_'.join([norm_label, str(xmin), str(xmax), str(ymin), str(ymax)]) + '_' + img_name
            save_path = os.path.join(save_right_dir, save_name) if norm_label == rec else os.path.join(save_error_dir, save_name)

            roi = img[ymin: ymax, xmin: xmax]
            cv2.imwrite(save_path, roi)
    print('==================== Done ====================\n')

def batchDet2Rec4LabelData(img_dir, save_dir, rec_item='CA', label_indx=-1, key_date='20181029', version='v0.0.0'):
    '''
    将原始数据进行定位及识别，并且将识别结果与标注一致的内容保存在model_right中供训练模型直接使用，
    不一致的保存在model_error中提交数据部门标注，
    提交地址：192.168.30.41/ocr/..
    user: hena / ocr
    password: 1q2w3e4r
    '''
    if not key_date:
        return
    img_type_dir = os.path.join(img_dir, rec_item, key_date)
    if not os.path.exists(img_type_dir):
        print("Warring:: rec_item/key_date[%s/%s] may be not ok! U need run bash data/download_shred_data.sh %s to download data!!!" %(rec_item, key_date, key_date))
        return
    
    for i, folder in enumerate(os.listdir(img_type_dir)):
        print("=====> Deal with %sth folder/date[%s/%s]"%(str(i+1), folder, key_date))
        image_dir = os.path.join(img_type_dir, folder)
        save_right_dir_date = os.path.join(save_dir, rec_item, 'model_right', key_date, folder)
        if not os.path.exists(save_right_dir_date):
            os.makedirs(save_right_dir_date)
        else:
            print("Warring: %s[%s/%s] exists! U may be had deal it" %(rec_item, key_date, folder))
        save_error_dir_date = os.path.join(save_dir, rec_item, 'model_error', key_date, folder)
        if not os.path.exists(save_error_dir_date):
            os.makedirs(save_error_dir_date)
        det2Rec4LabelData(image_dir, save_right_dir_date, save_error_dir_date, rec_item, label_indx, version)
    print('==================== Done ====================\n')

def main():
    args = check_label_args()
    tmp = args.key_date
    key_dates = [tmp] if ',' not in tmp else tmp.split(',') if type(tmp) == type('Hi') else tmp
    for key_date in key_dates:
        batchDet2Rec4LabelData(args.img_dir, args.save_dir, args.rec_item, args.label_indx, key_date, args.version)

if __name__ == '__main__':
    main()
