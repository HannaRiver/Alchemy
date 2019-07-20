import os
import shutil


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def pick_part_data4filelist(file_list_path='/work/competitions/TinyMind/Det/res0.1_part1.txt', save_dir='/work/competitions/TinyMind/Det/0.1part1/img', img_dir='/work/competitions/TinyMind/PreTrainData/0.1'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_list = readTxt(file_list_path)
    for img_name in file_list:
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(img_path, " not exists!")
            continue
        shutil.copy(img_path, os.path.join(save_dir, img_name))

if __name__ == '__main__':
    pick_part_data4filelist()


        