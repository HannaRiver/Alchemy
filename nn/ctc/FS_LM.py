#-*- coding=utf-8 -*-
import os
import sys
import argparse
import logging
import csv
import glob
import pickle
import numpy as np
from functools import reduce

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist


def strQ2B2(ustring):
    '''
    全角转半角，并且统一化”“-> ""
    '''
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        # 不要空格
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        tmp_segment = chr(inside_code)
        if tmp_segment in ['“', '”']:
            tmp_segment = '"'
        if tmp_segment in ['‘', '’']:
            tmp_segment = '\''
        if tmp_segment in ['—']:
            tmp_segment = '-'
        if tmp_segment in ['。']:
            tmp_segment = '.'
        if tmp_segment == ' ':
            continue
        rstring += tmp_segment
    return rstring

def ischn(astr):
    '''判断是否为中文
    '''
    new_str = ""
    for i in astr:
        if i in [',', '.', '-', ' ', '△', '$', '+', '*', '-', '(', ')', '~', '=', '#', '&', '、', '·']:
            continue
        new_str += i
    if new_str == "":
        return False
    return not new_str.isdigit()

def getLMMat(csv_list, cache_path='lmmat.pkl'):
    '''读取csv列表中的数据并进行统计
    先不考虑再已有的字典中添加数据这种情况
    '''
    # if os.path.exists(cache_path):
    #     allcnt, lmMat, classes = pickle.load(cache_path)
    # else:
    #     allcnt, lmMat, classes = 0, np.array([]), ''
    classes_list = []
    cnt_list = []
    lmMat = []
    for j, csv_file in enumerate(csv_list):
        # logging.info("[%s]csv file: %s " %(j+1, csv_file))
        with open(csv_file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            for row in readCSV:
                for item in row:
                    Bstr = strQ2B2(item)
                    if ischn(Bstr):
                        if len(list(set(['~', '+', 'S', 'Z', 'L']) & set(Bstr))) != 0:
                            continue
                        for i, seg in enumerate(Bstr):
                            if seg in ['=', '~', '+', '', '$', 'l', '?', 'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '\\']:
                                continue
                            if seg in ['。', 'S', 'Z', 'L', '増']:
                                logging.info("[%s]csv file: %s " %(j+1, csv_file))
                                logging.info("Warning:: %s -> %s" %(seg, Bstr))
                            if seg not in classes_list:
                                classes_list.append(seg)
                                cnt_list.append(1)
                                seg_idx = len(classes_list) - 1
                                lmMat.append([0]*len(classes_list)) # 初始化转移矩阵
                            else:
                                seg_idx = classes_list.index(seg)
                                cnt_list[seg_idx] += 1
                            if i == len(Bstr) - 1:
                                continue
                            next_seg = Bstr[i+1]
                            if next_seg not in classes_list:
                                lmMat[seg_idx].append(1)
                            else:
                                next_seg_idx = classes_list.index(next_seg)
                                lmMat[seg_idx] += [0]*(len(classes_list) - len(lmMat[seg_idx]))
                                lmMat[seg_idx][next_seg_idx] += 1
    lmMat = [(np.array(item + [0]*(len(classes_list) - len(item)))+ 1)/(cnt_list[i]+len(cnt_list)) for i, item in enumerate(lmMat)]
    # lmMat = [np.array(item + [0]*(len(classes_list) - len(item))) for item in lmMat]
    return  classes_list, cnt_list, lmMat

def dumo_pickel_data(csv_list, pkl_savepath):
    logging.info("countting data...")

    classes_list, cnt_list, lmMat = getLMMat(csv_list)

    logging.info("classes total: %s " %(len(classes_list)))
    logging.info("classes: %s " %(str(' ').join(i for i in classes_list)))
    logging.info("pickled save path: %s  pickling data..." %pkl_savepath)

    fw = open(pkl_savepath, 'wb')
    pickle.dump([classes_list, cnt_list, lmMat], fw)
    fw.close()

    logging.info("----------------D-O-N-E-------------------------")

def main():
    logging.info("========== Making LM Dict and Classes ==========")

    csv_file_path = '/work/hena/scripts/ocr/analysis/lm_csv_list.txt'
    pkl_savepath = '/work/hena/scripts/ocr/analysis/fs_lm_info.pkl'
    csv_list = readTxt(csv_file_path)

    logging.info("csv file total:: %s" %len(csv_list))

    dumo_pickel_data(csv_list, pkl_savepath)

if __name__ == '__main__':
    main()

                             






