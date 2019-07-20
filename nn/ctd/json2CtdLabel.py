#/usr/bin/python3
#-*- encoding=utf-8 -*-

import os
import json
import math
from functools import reduce

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

_IS_DEBUG_ = False

def distance4point(p1_x, p1_y, p2_x, p2_y):
    '''
    return: 两点之间的距离
    '''
    return math.sqrt((p1_x - p2_x)*(p1_x - p2_x) + (p1_y - p2_y)*(p1_y - p2_y))

def get_ctdlabel_bias(p_star, p_end):
    base_pw = abs(p_star[0] - p_end[0]) / 6
    base_ph = abs(p_star[1] - p_end[1]) / 6
    pw_list = [p_star[0] + (-1)**(p_star[0] > p_end[0]) * int((i) * base_pw) for i in range(7)]
    ph_list = [p_star[1] + (-1)**(p_star[1] > p_end[1]) * int((i) * base_ph) for i in range(7)]
    
    return pw_list, ph_list

def line2point4ctd(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    
    xmin = min(p1x, p2x, p3x, p4x)
    ymin = min(p1y, p2y, p3y, p4y)
    xmax = max(p1x, p2x, p3x, p4x)
    ymax = max(p1y, p2y, p3y, p4y)
    ctd_label_pre = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax)

    x_mean = (p1x + p2x + p3x + p4x) / 4
    y_mean = (p1y + p2y + p3y + p4y) / 4
    pointlist = [[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]]
    count_1, count_2, count_3, count_4 = 0, 0, 0, 0
    for item in pointlist:
        if item[0] < x_mean and item[1] < y_mean:
            p1_x = item[0]
            p1_y = item[1]
            count_1 = 1
            continue
        if item[0] < x_mean and item[1] > y_mean:
            p2_x = item[0]
            p2_y = item[1]
            count_2 = 1
            continue
        if item[0] > x_mean and item[1] > y_mean:
            p3_x = item[0]
            p3_y = item[1]
            count_3 = 1
            continue
        if item[0] > x_mean and item[1] < y_mean:
            p4_x = item[0]
            p4_y = item[1]
            count_4 = 1
            continue
    if count_1 + count_2 + count_3 + count_4 != 4:
        if _IS_DEBUG_ and 0:
            print("ERROR: image tooooooo bad!")
            print(str(list(map(str, [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]))))
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y = p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y

    pointlist = [[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]]
    distance_list = list(map(distance4point, [p1_x, p2_x, p3_x, p4_x], [p1_y, p2_y, p3_y, p4_y], [p2_x, p3_x, p4_x, p1_x], [p2_y, p3_y, p4_y, p1_y]))
    longest_index = distance_list.index(max(distance_list))

    if pointlist[longest_index -3][1] < pointlist[longest_index - 2][1]:
        ctd_p1 = pointlist[longest_index - 3]
        ctd_p2 = pointlist[longest_index]
        ctd_p3 = pointlist[longest_index - 1]
        ctd_p4 = pointlist[longest_index - 2]
    else:
        ctd_p1 = pointlist[longest_index - 1]
        ctd_p2 = pointlist[longest_index - 2]
        ctd_p3 = pointlist[longest_index - 3]
        ctd_p4 = pointlist[longest_index]

    # pw1to4, ph1to4 = [ctd_p1[0], ctd_p2[0], ctd_p3[0], ctd_p4[0]], [ctd_p1[1], ctd_p2[1], ctd_p3[1], ctd_p4[1]]
    pw1to7, ph1to7 = get_ctdlabel_bias(ctd_p1, ctd_p2)
    pw8to14, ph8to14 = get_ctdlabel_bias(ctd_p3, ctd_p4)
    tmp_pwh1to14 = list(map(lambda x, y: ',' + str(x - xmin) + ',' + str(y - ymin), pw1to7 + pw8to14, ph1to7 + ph8to14))
    ctd_label = reduce(lambda x, y: x + y, [ctd_label_pre] + tmp_pwh1to14)

    return ctd_label

def json2CTDlabel(josn_path, save_lable_path):
    # print(save_lable_path + " Savng...")
    fw = open(save_lable_path, 'w')
    with open(josn_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for i in obj:
            point_list = i['polygon']
            if _IS_DEBUG_:
                if len(point_list) != 4:
                    print("Label Error:: " + str(point_list) + '\n')
            p1_x, p2_x, p3_x, p4_x = list(point_list[i][0] for i in range(4))
            p1_y, p2_y, p3_y, p4_y = list(point_list[i][1] for i in range(4))
            fw.write(line2point4ctd(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y) + '\n')
    fw.close()

def batch_json2CTDlabel():
    json_dir = "/work/hena/ocr/data/Paperwork/DrivingLicense/json/"
    img_dir = "/work/hena/ocr/data/Paperwork/DrivingLicense/mark/"
    save_label_dir = "/work/hena/ocr/detector/CTD/data/ctw1500/train/text_label_curve/"
    img_list = os.listdir(img_dir)
    
    logging.info("==================== Point ROI(json) --> CTD 32 Label(txt) ====================")
    logging.info("Json Dir: %s" %(json_dir))
    logging.info("Img Dir: %s" %(img_dir))
    logging.info("Img Size: %s" %(len(img_list)))
    logging.info("Save Dir: %s" %(save_label_dir))

    for item in img_list:
        json_path = os.path.join(json_dir, item + '.json')
        save_label_path = os.path.join(save_label_dir, item[: -3] + 'txt')
        json2CTDlabel(json_path, save_label_path)
    
    logging.info("================= Org2CTD Label DONE! ===============")

if __name__ == '__main__':
    batch_json2CTDlabel()