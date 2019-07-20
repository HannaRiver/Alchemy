#/usr/bin/python
#-*- encoding=utf-8 -*-
import os
import sys
import argparse
import shutil
import cv2
from alchemy_config import cfg, cfg_from_file
sys.path.append(cfg.UTILS_DIR)
sys.path.append(os.path.join(cfg.CRCB_DIR, 'utils'))
from file2list import readTxt, get_label4name
from shred_recognise import get_recognition, refuse2plate
from resize_img import undeform_center_resize
from resize_img import map4undeform_center_resize
from lca2ca import cncurrency
from norm_shred_label import normShredLabel
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

RecItem2Cfg = {
    'CA': cfg.CRCB_CFG_NAME.CA_PRINT,
    'LA': cfg.CRCB_CFG_NAME.LA_PRINT,
    'CD': cfg.CRCB_CFG_NAME.CD_PRINT,
    'LD': cfg.CRCB_CFG_NAME.LD_PRINT,
    'Chn': cfg.CRCB_CFG_NAME.CHN_PRINT,
    'Num': cfg.CRCB_CFG_NAME.NUM_PRINT,
}

def shred_rec_args():
    '''
    常熟碎片识别模型参数(大写金额类-打印体)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default=cfg.CRCB_DATA.CHECKDATA_DIR_PRINT, help='需要处理数据的地址')
    parser.add_argument('--label_indx', type=int, default=0, help='碎片数据gt的位置')
    parser.add_argument('--key_date', default=cfg.CRCB_DATA.KEYDATE, type=str, help='需要处理的日期批次号')
    parser.add_argument('--version', default='v0.0.0', type=str, help='识别模型的版本号')
    parser.add_argument('--rec_item', default=cfg.CRCB_DATA.RECITEM, help='处理的碎片大类[CA, LA, CD, LD, Num, Chn, Other]', type=str)
    parser.add_argument('--debug_rec', default=False, help='是否看识别的具体结果', type=bool)
    parser.add_argument('--showkey', default=False, help='是否看图', type=bool)

    parser.add_argument('--img_path', default='')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--model_def', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_MODELDEF)
    parser.add_argument('--image_resize', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_RESIZESIZE, type=int)
    parser.add_argument('--time_step', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_TIMESTEP)
    parser.add_argument('--scale', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_SCALE)
    parser.add_argument('--resize_type', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_RESIZETYPE)
    parser.add_argument('--mean_value', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_MEANVALUE)
    parser.add_argument('--model_weights', default=cfg.CRCB_MODEL.SHRED_CA_PRINT_REC_MODELWEIGHT)
    parser.add_argument('--cfg_path', default='')
    return parser.parse_args()

def CheckLabelRec(rec_item, recognition, img_dir, save_right_dir, save_error_dir, label_indx=0, _IS_DEBUG_REC_=False, showkey=False, version='V0.0.0'):
    if not os.path.exists(save_right_dir):
        os.makedirs(save_right_dir)
    if not os.path.exists(save_error_dir):
        os.makedirs(save_error_dir)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        label = get_label4name(os.path.splitext(img_name)[0], label_indx, '_')
        img = cv2.imread(img_path)
        pre_result, _, _, = refuse2plate(img, recognition, _IS_DEBUG_REC_)
        norm_label = normShredLabel(label, rec_item, pre_result)
        save_name = norm_label + '_' + pre_result + '_' + img_name
        save_path = os.path.join(save_right_dir, save_name) if norm_label == pre_result else os.path.join(save_error_dir, save_name)
        shutil.copy(img_path, save_path)

def BatchCheckLabelRec(rec_item, img_root, save_right_root, save_error_root, label_indx=0, _IS_DEBUG_REC_=False, showkey=False, version='V0.0.0'):
    rec_args = shred_rec_args()
    recognition = get_recognition(rec_args) # 获取lstm识别器
    for folder in os.listdir(img_root):
        logging.info("-----> Deal with the folder %s" %(folder))
        img_dir = os.path.join(img_root, folder)
        save_right_dir = os.path.join(save_right_root, folder)
        save_error_dir = os.path.join(save_error_root, folder)
        CheckLabelRec(rec_item, recognition, img_dir, save_right_dir, save_error_dir, label_indx, _IS_DEBUG_REC_, showkey, version)

def run(args, key_date, rec_item):
    img_root = os.path.join(args.img_dir, rec_item, 'return', key_date)
    save_right_root = os.path.join(args.img_dir, rec_item, 'check_right', key_date)
    save_error_root = os.path.join(args.img_dir, rec_item, 'check_error', key_date)
    logging.info("==================== Check Labeled Rec Data By Model =====================")
    logging.info("Model Version: %s \t Model Name: %s" %(args.version, os.path.basename(args.model_weights)))
    logging.info("Mean Value: %s \t Scale: %s \t Resize: %s \t Resize Type: %s"  %(args.mean_value, args.scale, str(args.image_resize), args.resize_type))
    logging.info("处理的文件目录为: %s" %(img_root))
    logging.info("Model Right Root: %s" %(save_right_root))
    logging.info("Model Error Root: %s" %(save_error_root))
    BatchCheckLabelRec(rec_item, img_root, save_right_root, save_error_root, args.label_indx, args.debug_rec, args.showkey, args.version)
    logging.info("==================== Check Labeled Rec Data By Model Done =====================")

def main():
    args = shred_rec_args()
    rec_item = args.rec_item
    cfg_file = RecItem2Cfg[rec_item]
    cfg_path = args.cfg_path if args.cfg_path else osp.join(cfg.CRCB_DIR, 'rec', cfg_file)
    cfg_from_file(cfg_path)
    tmp = args.key_date
    key_dates = [tmp] if ',' not in tmp else tmp.split(',') if type(tmp) == type('Hi') else tmp
    for key_date in key_dates:
        run(args, key_date, rec_item)

if __name__ == '__main__':
    main()