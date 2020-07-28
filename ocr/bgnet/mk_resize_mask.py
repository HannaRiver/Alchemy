#/usr/bin/python3
#-*- encoding=utf-8 -*-

import os
import json
import cv2
import numpy as np


def mk_resize_mask(mark_path, json_path, save_mark_path, save_mask_path, resize_h=512, resize_w=512):
    '''
    根据json画mask图 --> mask.bmp
    '''
    img = cv2.imread(mark_path)
    img_h, img_w = img.shape[: 2]
    ratio_h = resize_h / img_h
    ratio_w = resize_w / img_w

    resize_mark = cv2.resize(img, (resize_w, resize_h))
    cv2.imwrite(save_mark_path, resize_mark)

    mask = np.zeros((resize_h, resize_w))

    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for i in obj:
            for j in range(len(i['path']) - 1):
                # 横线
                if i['label'] == 'A':
                    star_x = int(i['path'][j][0] * ratio_w)
                    star_y = i['path'][j][1]
                    end_y = i['path'][j + 1][1]
                    end_x = int(i['path'][j + 1][0] * ratio_w)
                    label_y = int((star_y + end_y)/2 * ratio_h)
                    star_y, end_y = label_y, label_y

                    cv2.line(mask, (star_x, star_y), (end_x, end_y), 1, 1)

                    continue

                # 竖线
                if i['label'] == 'B':
                    star_x = i['path'][j][0]
                    star_y = int(i['path'][j][1] * ratio_h)
                    end_y = int(i['path'][j + 1][1] * ratio_h)
                    end_x = i['path'][j + 1][0]
                    label_x = int((star_x + end_x)/2 * ratio_w)
                    star_x, end_x = label_x, label_x

                    cv2.line(mask, (star_x, star_y), (end_x, end_y), 2, 1)

                    continue
                
                else:
                    print("Tag ERROR: " + mark_path + " label :: " + i['label'])
                    continue
    
    cv2.imwrite(save_mask_path, mask)

def batch_mk_mask(labeled_dir, save_mark_dir, save_mask_dir):
    assert(os.path.exists(labeled_dir)), labeled_dir + " not exists!"
    if not os.path.exists(save_mark_dir):
        os.mkdir(save_mark_dir)
    if not os.path.exists(save_mask_dir):
        os.mkdir(save_mask_dir)
    
    img_list = os.listdir(os.path.join(labeled_dir, 'mark'))

    for item in img_list:
        mark_path = os.path.join(labeled_dir, 'mark', item)
        json_path = os.path.join(labeled_dir, 'json', item + '.json')
        save_mark_path = os.path.join(save_mark_dir, item[: -4] + '.bmp')
        save_mask_path = os.path.join(save_mask_dir, item[: -4] + '.bmp')

        if not os.path.exists(mark_path) or not os.path.exists(json_path):
            print(item, " not exists!!!")
            continue
        mk_resize_mask(mark_path, json_path, save_mark_path, save_mask_path)

if __name__ == '__main__':
    labeled_dir = '/work/hena/ocr/data/TableLine/CamVid_seg/6_100_done'
    save_mark_dir = '/work/hena/ocr/data/TableLine/CamVid_seg/trian'
    save_mask_dir = '/work/hena/ocr/data/TableLine/CamVid_seg/trianannot'
    batch_mk_mask(labeled_dir, save_mark_dir, save_mask_dir)


                    





