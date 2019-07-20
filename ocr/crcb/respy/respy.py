# -*- coding: utf-8 -*-
import pickle
import cv2
import os
import numpy as np

def respy(caffemodel, save_path):
    fr = open(caffemodel, 'rb')
    all_data = pickle.load(fr)
    for item in all_data:
        img = item[0]
        name = item[1]
        # savename = img2savename[name]
        
        cv2.imwrite(save_path + name + '.png', img)

def respy2label(caffemodel, save_path):
    fw = open(save_path, 'w')
    fr = open(caffemodel, 'rb')
    all_data = pickle.load(fr)
    for item in all_data:
        img_name = item[0]
        label = item[1]
        item_name = item[2][1:]
        fw.write(img_name + ',' + label + ',' + item_name + '\n')
    fw.close()

        

if __name__ == '__main__':
    # root_dir = '/work/hena/ocr/data/crcb/spy/0711/caffemodel_0712_all/'
    # save_dir = '/work/hena/ocr/data/crcb/img/0711_4/'
    # file_list = os.listdir(root_dir)
    # for i in file_list:
    #     respy(root_dir + i, save_dir)
    # print("respy done!")

    # make label file
    respy2label('./CRCB/crcb_shred/spy/0712/train_label_03.caffemodel', './CRCB/crcb_shred/spy/CRCB_0712.txt')
    pass

