#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import os.path as osp
import sys
from alchemy_config import cfg, cfg_from_file
sys.path.append(cfg.UTILS_DIR)
from file2list import readTxt
import pickle
import argparse


RecItem2Cfg = {
    'CA': cfg.CRCB_CFG_NAME.CA_PRINT,
    'LA': cfg.CRCB_CFG_NAME.LA_PRINT,
    'CD': cfg.CRCB_CFG_NAME.CD_PRINT,
    'LD': cfg.CRCB_CFG_NAME.LD_PRINT,
    'Chn': cfg.CRCB_CFG_NAME.CHN_PRINT,
    'Num': cfg.CRCB_CFG_NAME.NUM_PRINT,
}

RecItem2LabelList = {
    'CA': cfg.CRCB_WORDSLIB.CA_LIST,
    'CD': cfg.CRCB_WORDSLIB.CD_LIST,
}

RecItem2LabelNummax = {
    'CA': cfg.CRCB_WORDSLIB.CA_NUMMAX,
    'CD': cfg.CRCB_WORDSLIB.CD_NUMMAX
}


def shred_labeld_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_indx', type=int, default=0, help='碎片数据gt的位置')
    parser.add_argument('--rec_item', default=cfg.CRCB_DATA.RECITEM, help='处理的碎片大类[CA, LA, CD, LD, Num, Chn, Other]', type=str)
    parser.add_argument('--cfg_path', default='')
    parser.add_argument('--img_txt', default=cfg.CRCB_WORDSLIB.IMG_TXT)
    parser.add_argument('--labeled_path', default=cfg.CRCB_WORDSLIB.LABELED_TXT)
    return parser.parse_args()

def imgname2label(name_list_path, savepath, num_tab, nummax, indx=0):
    new_num_tab = num_tab
    black_id = num_tab.index(' ')

    img_name_list = readTxt(name_list_path)

    fw = open(savepath, 'w')

    for item in img_name_list:
        # print(item)
        label = os.path.splitext(os.path.basename(item))[0].split('_')[indx]
        cnt = 0
        for i in label:
            if i not in num_tab:
                new_num_tab.append(i)
                print(i)
            cnt += 1
            if cnt == 1:
                fw.write(str(new_num_tab.index(i)))
                continue
            fw.write(' ' + str(new_num_tab.index(i)))
        fw.write((' '+str(black_id)) * (nummax - cnt) + '\n')
    fw.close()
    return new_num_tab

if __name__ == '__main__':
    args = shred_labeld_args()
    rec_item = args.rec_item
    cfg_file = RecItem2Cfg[rec_item]
    cfg_path = args.cfg_path if args.cfg_path else osp.join(cfg.CRCB_DIR, 'rec', cfg_file)
    cfg_from_file(cfg_path)
    combi_savepath = args.labeled_path
    name_list_path = args.img_txt
    label_indx = args.label_indx
    num_tab = RecItem2LabelList[rec_item]
    nummax = RecItem2LabelNummax[rec_item]
    new_num_tab = imgname2label(name_list_path, savepath, num_tab, nummax, label_indx)
    os.system("paste -d' ' %s %s >> %s" %(name_list_path, label_savepath, combi_savepath))
    fw1 = open(os.path.join(cfg.CRCB_DIR, 'wordslib', rec_item.lower()+'_wordslib3.pkl'), 'wb')
    print(len(new_num_tab))
    pickle.dump(new_num_tab, fw1)
    fw1.close()
