#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import sys
import csv
import shutil
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from str2info import get_label4name


def get_platename_by_csvfile(csv_path):
    plate_dit = {}
    with open(csv_path) as csvfile:
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            hphm = row[0]
            fzjg = row[4]
            plate_dit[hphm] = fzjg
    return plate_dit

def batch_newenergy_rename(img_dir, csv_path, save_dir):
    plate_dit = get_platename_by_csvfile(csv_path)
    for item in os.listdir(img_dir):
        hphm = get_label4name(item, 1, '_')
        if hphm not in plate_dit:
            print(item)
            continue
        fzjg = plate_dit[hphm]
        plate = fzjg + hphm[1: ]

        img_path = os.path.join(img_dir, item)
        save_path = os.path.join(save_dir, plate + '_' + item)

        shutil.copy(img_path, save_path)




if __name__ == '__main__':
    csv_path = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/hphm_test.csv'
    img_dir = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/suzhou/pick_newenergy_error'
    save_dir = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/suzhou/pick_newenergy_error_rename'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    batch_newenergy_rename(img_dir, csv_path, save_dir)
    