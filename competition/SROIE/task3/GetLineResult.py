#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import numpy as np
import cv2
import shutil
import shapely
from shapely.geometry import Polygon,MultiPoint


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            info_list = line.strip().split(',')
            locates = [info_list[0]] + [int(i) for i in info_list[1: 9]]
            locate = ','.join(info_list[: 9])
            content = line.strip()[len(locate)+1: ]
            locates.append(content)
            filelist.append(locates)
    return filelist

def readTxt_NoneResult(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            info_list = ['O'] + line.strip().split(',')
            locates = ['O'] + [int(i) for i in info_list[1: 9]]
            locate = ','.join(info_list[1: 9])
            content = line.strip()[len(locate)+1: ]
            locates.append(content)
            filelist.append(locates)
    return filelist


def IntersectBBox(bbox1, bbox2):
    '''
    bbox : [xmin, ymin, xmax, ymax]
    '''
    if bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1]:
        return [0, 0, 0, 0]
    return [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])]

def IOU(line1, line2):
    '''
    line1/2: [x1, y1, x2, y2, x3, y3, x4, y4]
    '''
    a = np.array(line1).reshape(4, 2)
    b = np.array(line2).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))
    if not poly1.intersects(poly2):
        return 0
    else:
        inter_area = poly1.intersection(poly2).area
        # union_area = MultiPoint(union_poly).convex_hull.area
        union_area = poly1.area
        return 0 if union_area == 0 else float(inter_area) / union_area

def getLineResult(txtpath):
    """
    [
        [],
        [],
        [],
    ]
    """
    #######tmp#######
    # img = cv2.imread('/work/competitions/ICDAR/SROIE/data/task2_train/X51005268200.jpg')
    ##############
    lineMat = [] # 按行排序
    # itemlist = readTxt(txtpath)
    itemlist = readTxt_NoneResult(txtpath)
    itemlist.sort(key=lambda x: (min([x[i] for i in [2, 4, 6, 8]]), min([x[i] for i in [1, 3, 5, 7]])))
    line_list = [itemlist[0]]
    # 获取第一个区域的信息
    first_ymin = min([itemlist[0][i] for i in [2, 4, 6, 8]])
    first_ymax = max([itemlist[0][i] for i in [2, 4, 6, 8]])
    first_xmin = min([itemlist[0][i] for i in [1, 3, 5, 7]])
    first_xmax = max([itemlist[0][i] for i in [1, 3, 5, 7]])
    first_w = max(first_xmax - first_xmin, 0)
    first_h = max(first_ymax - first_ymin, 0)
    first_center_y = sum([itemlist[0][i] for i in [2, 4, 6, 8]])/4
    line1 = itemlist[0][1: 9]
    for i, item in enumerate(itemlist[1: ]):
        bbox_xmin = min([item[i] for i in [1, 3, 5, 7]])
        bbox_xmax = max([item[i] for i in [1, 3, 5, 7]])
        bbox_ymin = min([item[i] for i in [2, 4, 6, 8]])
        bbox_ymax = max([item[i] for i in [2, 4, 6, 8]])
        bbox_h = max(bbox_ymax - bbox_ymin, 0)
        bbox_w = max(bbox_xmax - bbox_xmin, 0)
        bbox_center_y = sum([item[i] for i in [2, 4, 6, 8]])/4
        ###############################
        # cv2.rectangle(img, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), (0, 0, 255), 1)
        # cv2.namedWindow("line sort result", cv2.WINDOW_NORMAL)
        # cv2.imshow("line sort result", img)
        # cv2.waitKey(0)
        ###############################
        # 如果IOU > 0.1另外一行
        line2 = item[1: 9]
        intersectbbox = IntersectBBox([first_xmin, first_ymin, first_xmax, first_ymax], [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        w, h = intersectbbox[2] - intersectbbox[0], intersectbbox[3] - intersectbbox[1]
        # iou = IOU(line1, line2)
        iou = 0 if intersectbbox==[0, 0, 0, 0] else float(w*h)/(first_h*first_w)
        # 认为如果下个比上一个的差距差一半高度以上则另起一行
        if bbox_center_y - first_center_y < first_h/2 and iou < 0.1:
            # first_h, first_w = bbox_h, bbox_w
            # first_center_y = bbox_center_y
            # first_xmax, first_ymax = bbox_xmax, bbox_ymax
            line_list.append(item)
        else:
            first_h, first_w = bbox_h, bbox_w
            first_center_y = bbox_center_y
            first_xmin, first_ymin = bbox_xmin, bbox_ymin
            first_xmax, first_ymax = bbox_xmax, bbox_ymax
            lineMat.append(line_list)
            line_list = [item]
            #######################################################
            # bbox_xmin = min([min([j[i] for i in [1, 3, 5, 7]]) for j in lineMat[-1]])
            # bbox_xmax = max([max([j[i] for i in [1, 3, 5, 7]]) for j in lineMat[-1]])
            # bbox_ymin = min([min([j[i] for i in [2, 4, 6, 8]]) for j in lineMat[-1]])
            # bbox_ymax = max([max([j[i] for i in [2, 4, 6, 8]]) for j in lineMat[-1]])
            # cv2.rectangle(img, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), (255, 0, 0), 3)
            # cv2.namedWindow("line sort result", cv2.WINDOW_NORMAL)
            # cv2.imshow("line sort result", img)
            # cv2.waitKey(0)
            ########################################################
    lineMat.append(line_list)
    return lineMat

def LineSortResult(linelist):
    linelist.sort(key=lambda x: (min([x[i] for i in [1, 3, 5, 7]]), min([x[i] for i in [2, 4, 6, 8]])))
    return linelist

def getImgLineSortResult(txtpath):
    lineMat = getLineResult(txtpath)
    return [LineSortResult(linelist) for linelist in lineMat]

def LineSortResult2Txt(lineMat, save_path):
    fw = open(save_path, 'w')
    for line in lineMat:
        for item in line:
            fw.write(','.join([str(i) for i in item]) + '\n')
    fw.close()
    

def TestGetLineResult(img_path, txtpath):
    img = cv2.imread(img_path)
    lineMat = getLineResult(txtpath)
    for i, item in enumerate(lineMat):
        for x in item:
            color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            ymin = min(x[2], x[4], x[6], x[8])
            ymax = max(x[2], x[4], x[6], x[8])
            xmin = min(x[1], x[3], x[5], x[7])
            xmax = max(x[1], x[3], x[5], x[7])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(img, (str(i+1)), (xmin - 5, ymin - 5), font, 0.6, (0, 255, 0), 1)
    cv2.namedWindow("line sort result", cv2.WINDOW_NORMAL)
    cv2.imshow("line sort result", img)
    cv2.waitKey(0)

def TestGetLineSortResult(img_path, lineMat, save_path=''):
    img = cv2.imread(img_path)
    for i, item in enumerate(lineMat):
        xmin_, ymin_, xmax_, ymax_ = 100000000000, 1000000000000000000, 0, 0
        color_1 = (255, 0, 0) if i %2 == 0 else (0, 255, 0)
        for j, x in enumerate(item):
            color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            ymin = min(x[2], x[4], x[6], x[8])
            ymax = max(x[2], x[4], x[6], x[8])
            xmin = min(x[1], x[3], x[5], x[7])
            xmax = max(x[1], x[3], x[5], x[7])
            xmin_ = min(xmin_, xmin)
            ymin_ = min(ymin_, ymin)
            xmax_ = max(xmax_, xmax)
            ymax_ = max(ymax_, ymax)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(img, (str(i+1) + '-' + str(j+1)), (xmin - 5, ymin - 5), font, 0.6, (0, 255, 0), 1)
        cv2.rectangle(img, (xmin_, ymin_), (xmax_, ymax_), color_1, 1)
    # cv2.namedWindow("Img sort result", cv2.WINDOW_NORMAL)
    # cv2.imshow("Img sort result", img)
    # cv2.waitKey(0)
    # save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, img)

def BatchGetImgLineSortResult4Txt(txt_dir, img_dir, save_txt_dir='', save_img_dir=''):
    if save_img_dir:
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
    else:
        save_img_dir = os.path.join(txt_dir, '..', 'sort_img')
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
    if save_txt_dir:
        if not os.path.exists(save_txt_dir):
            os.mkdir(save_txt_dir)
    else:
        save_txt_dir = os.path.join(txt_dir, '..', 'sort_txt')
        if not os.path.exists(save_txt_dir):
            os.mkdir(save_txt_dir)
    
    for item in os.listdir(txt_dir):
        img_name = os.path.splitext(item)[0] + '.jpg'
        txt_path = os.path.join(txt_dir, item)
        img_path = os.path.join(img_dir, img_name)
        save_img_path = os.path.join(save_img_dir, img_name)
        save_txt_path = os.path.join(save_txt_dir, item)

        lineMat = getImgLineSortResult(txt_path)
        LineSortResult2Txt(lineMat, save_txt_path)
        TestGetLineSortResult(img_path, lineMat, save_img_path)


def main():
    txt_dir = '/work/competitions/ICDAR/SROIE/task3/recogResult_0428'
    img_dir = '/work/competitions/ICDAR/SROIE/data/test/task1_2_test'
    BatchGetImgLineSortResult4Txt(txt_dir, img_dir)

def get_right_rm():
    image_dir = '/work/competitions/ICDAR/SROIE/task3/sort_img'
    txt_dir = '/work/competitions/ICDAR/SROIE/task3/sort_txt'
    save_dir = '/work/competitions/ICDAR/SROIE/task3/sort_txt_ok'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for item in os.listdir(image_dir):
        txt_path = os.path.join(txt_dir, os.path.splitext(item)[0] + '.txt')
        save_path = os.path.join(save_dir, os.path.splitext(item)[0] + '.txt')
        shutil.move(txt_path, save_path)
        


if __name__ == '__main__':
    main()