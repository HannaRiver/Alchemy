#/usr/bin/python3
#-*- encoding=utf-8 -*-
'''
获取任意四边形旋转过后的roi
'''
import cv2
import numpy as np
import math


def get_quad_roi(quads, img):
    '''
    quads: [x1, y1, x2, y2, x3, y3, x4, y4]
    img: orig image
    -> rot roi
    '''
    quads = [int(math.ceil(float(i))) for i in quads]
    cnt = np.array([[quads[0],quads[1]], [quads[2],quads[3]], [quads[4],quads[5]], [quads[6],quads[7]]])
    rect = cv2.minAreaRect(cnt)
    rect_w, rect_h = rect[1]
    theta = rect[2]
    if theta <= -45:
        rect_h, rect_w = rect[1]
        theta = -90 - theta

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    xmin = min([box[i][0] for i in range(4)])
    xmax = max([box[i][0] for i in range(4)])
    ymin = min([box[i][1] for i in range(4)])
    ymax = max([box[i][1] for i in range(4)])

    center_x, center_y = (xmax + xmin)/2.0, (ymax + ymin)/2.0
    w = max(xmax - xmin, rect_w)
    h = max(ymax - ymin, rect_h)

    xmin = int(center_x - w/2)
    ymin = int(center_y - h/2)
    xmax = int(center_x + w/2)
    ymax = int(center_y + h/2)

    big_roi = img[ymin: ymax, xmin: xmax]
    rows, cols =big_roi.shape[: 2]
    cols1 = int(rect_w)

    M = cv2.getRotationMatrix2D((cols/2.0,rows/2.0), theta, 1)
    dst = cv2.warpAffine(big_roi, M, (int(cols1), int(rows)))
    
    new_xmin = 0
    new_ymin = int(rows/2.0 - rect_h/2.0)
    new_xmax = cols1
    new_ymax = int(rows/2.0 + rect_h/2.0)
    new_xmin, new_ymin, new_xmax, new_ymax

    new_roi = dst[new_ymin: new_ymax, new_xmin: new_xmax]
    return new_roi

