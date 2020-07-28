#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import sys
import csv
import shutil
import glob
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from str2info import get_label4name


def get_platename_by_csvfile(csv_path):
    plate_dit = {}
    with open(csv_path) as csvfile:
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            hphm = row[0]
            fzjg = row[5]
            if fzjg == 'NULL' or fzjg == '无数据':
                continue
            plate_dit[hphm] = fzjg
    return plate_dit

def pic_newenergy_plate(img_root, csv_path, save_dir):
    enpty_cnt, en_cnt, other_plt = 0, 0, 0
    save_nonnewenergy_dir = os.path.join(save_dir, '..', 'other_ori')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_nonnewenergy_dir):
        os.mkdir(save_nonnewenergy_dir)
    plate_dit = get_platename_by_csvfile(csv_path)
    for hphm in plate_dit:
        fzjg = plate_dit[hphm]
        plate = fzjg + hphm[1: ]
        img_path_list = glob.glob(os.path.join(img_root, '*_' + hphm +'_*'))
        if len(img_path_list) == 0:
            enpty_cnt += 1
            continue
        for img_path in img_path_list:
            if len(hphm) == 7:
                en_cnt += 1
                save_path = os.path.join(save_dir, plate + '_' + os.path.basename(img_path))
                shutil.copy(img_path, save_path)
            else:
                other_plt += 1
                shutil.copy(img_path, os.path.join(save_nonnewenergy_dir, plate + '_' + os.path.basename(img_path)))
    print(enpty_cnt, en_cnt, other_plt)


if __name__ == '__main__':
    img_root = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/shenzhen/pic/all_ori'
    csv_path = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/shenzhen/0399.csv'
    save_dir = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/shenzhen/pick_newenergy'
    pic_newenergy_plate(img_root, csv_path, save_dir)