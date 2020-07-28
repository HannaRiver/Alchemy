#/usr/bin/python3
#-*- coding=utf-8 -*-

import sys
import os
import logging
import random
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
import cv2
import write_xml as wr_xml


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def getInfo(txtpath):
    result = {}
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            infos = line.strip().split(' ')
            if len(infos) != 9:
                print(line + " not ok!")
                continue
            img_name = infos[0]
            locate = [int(i) for i in infos[1: ]]
            result[img_name] = locate
    return result

def get_locate_img_xml(img_path, locate_rect, img_save_dir, xml_save_dir):
    img = cv2.imread(img_path)
    h, w, d = img.shape
    filename = os.path.basename(img_path)
    name = "id"
    bnbox = locate_rect

    xmin, ymin, xmax, ymax = [int(i) for i in bnbox]
    if ymin*xmin*xmax*ymax == 0:
        print(img_path, locate_rect)
        return
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    img_save_path = os.path.join(img_save_dir, filename)
    xml_save_path = os.path.join(xml_save_dir, filename.split('.')[0]+'.xml')
    wr_xml.writeInfo2Xml(xml_save_path, filename, name, xmin, ymin, xmax, ymax, w, h, d)
    cv2.imwrite(img_save_path, img)

def ploy2rect(ploys):
    xmin = min([ploys[i] for i in [0, 2, 4, 6]])
    xmax = max([ploys[i] for i in [0, 2, 4, 6]])
    ymin = min([ploys[i] for i in [1, 3, 5, 7]])
    ymax = max([ploys[i] for i in [1, 3, 5, 7]])
    return xmin, ymin, xmax, ymax

def batch_get_locate_img_xml(img_dir, locate_path, img_save_dir, xml_save_dir):
    '''
    根据locate_path中的定位结果写xml
    '''
    img_name2locate_dict = getInfo(locate_path)
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        if img_name in img_name2locate_dict:
            locate_poly = img_name2locate_dict[img_name]
        else:
            print(img_name + " not in Dict")
            continue
        locate_rect = ploy2rect(locate_poly)
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            logging.info('None File :: %s' %(img_path))
            continue        
        get_locate_img_xml(img_path, locate_rect, img_save_dir, xml_save_dir)

    


if __name__ == "__main__":
    # ==================================== batch1 ======================================
    img_save_dir = '/work/competitions/TinyMind/Det/0.1part1/check'
    xml_save_dir = '/work/competitions/TinyMind/Det/0.1part1/xml'
    img_dir = '/work/competitions/TinyMind/Det/0.1part1/img'
    locate_path = '/work/competitions/TinyMind/Det/result_0.1.txt'

    batch_get_locate_img_xml(img_dir, locate_path, img_save_dir, xml_save_dir)

