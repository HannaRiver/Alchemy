import sys
import os
from file2list import read_plate_vir_csv


def txt2download(txt_path, save_dir):
    info_list = read_plate_vir_csv(txt_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for item in info_list:
        os.system('aria2c -d %s/ %s' % (save_dir, item))

if __name__ == '__main__':
    txt_path = '/home/hena/下载/201901162016_huzhou_plate_problem.csv.train'
    save_dir = '/home/hena/下载/test'
    txt2download(txt_path, save_dir)

    
    
