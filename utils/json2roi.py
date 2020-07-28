#/usr/bin/python3
#-*- coding=utf-8 -*-

import os
import json
import cv2


def clpr_json2plate_roi(json_path):
    clpr_det, clpr_rec = [], []
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for item in obj:
            clpr_rect = obj['clpr']['plate_obj']['rect']
            roi_x, roi_y, roi_w, roi_h = [int(i) for i in clpr_rect]
            if roi_w <= 3 or roi_h <=3:
                print("ROI Json Warning::", json_path)
            else:
                clpr_det.append([roi_x, roi_y, roi_w, roi_h])
                plate_rec = obj['clpr']['plate_rec']['plate_num_gt']
                clpr_rec.append(plate_rec)
    return clpr_det, clpr_rec

def json2roi(json_path):
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for item in obj:
            roi_x, roi_y, roi_w, roi_h = [int(i) for i in item['rect']]
            if roi_w <= 3 or roi_h <=3:
                print("ROI Json Warning::", json_path)
            else:
                return roi_x, roi_y, roi_w, roi_h

    return roi_x, roi_y, roi_w, roi_h

def json2allroi(json_path):
    result = []
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for item in obj:
            roi_x, roi_y, roi_w, roi_h = [int(i) for i in item['rect']]
            if roi_w <= 3 or roi_h <=3:
                print("ROI Json Warning::", json_path)
            else:
                result.append([roi_x, roi_y, roi_w, roi_h])

    return result  

def get_chn_num(img_path, json_path, save_roi_path):
    '''
    保存roi
    '''
    img = cv2.imread(img_path)
    roi_x, roi_y, roi_w, roi_h = json2roi(json_path)
    roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
    cv2.imwrite(save_roi_path, roi_img)

def ploygon2rect(polygon):
    x_min, y_min = [int(min([item[i] for item in polygon])) for i in range(2)]
    x_max, y_max = [int(max([item[i] for item in polygon])) for i in range(2)]
    return x_min, y_min, x_max - x_min, y_max - y_min

def json2point(json_path,label='danweihuikuan'):
    if not os.path.exists(json_path):
        return False
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for item in obj:
            if item['label'] != label:
                continue
            point = [int(i) for i in item['polygon']]
            if len(point) != 4:
                print("标注的坐标点数目不对", json_path)
            point1, point2, point3, point4 = point

            roi_x, roi_y, roi_w, roi_h = ploygon2rect(item['polygon'])
    

def json2ctd_roi(img_path, json_path, save_path, label='danweihuikuan'):
    img = cv2.imread(img_path)
    if not os.path.exists(json_path):
        return
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for item in obj:
            if item['label'] != label:
                continue
            roi_x, roi_y, roi_w, roi_h = ploygon2rect(item['polygon'])
            if roi_w < roi_h:
                pass
            roi_img = img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
            cv2.imwrite(save_path, roi_img)

def main():
    save_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch6/roi_rengong'
    img_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch6/roi_batch6'
    json_dir = '/work/hena/ocr/pro/CLPR/generate_dir/batch6/json'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for item in os.listdir(json_dir):
        json_path = os.path.join(json_dir, item)
        save_path = os.path.join(save_dir, item[: -5])
        img_path = os.path.join(img_dir, item[: -5])

        json2ctd_roi(img_path, json_path, save_path)


if __name__ == "__main__":
    main()

