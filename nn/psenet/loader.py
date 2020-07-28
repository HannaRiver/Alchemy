import cv2
import numpy as np
import os
import Polygon as plg
import pyclipper


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        print(rate, (int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

def shrink_i(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

def shrink_my(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        max_shr = max(min_distance(bbox), 4)
        print(rate, (int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

def min_distance(bbox):
    min_dis = 100000000
    for i in range(bbox.shape[0]):
        tmp_dis = dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])*3/8
        # print('tmp_dis:',tmp_dis)
        min_dis = tmp_dis if tmp_dis < min_dis else min_dis
    return min_dis

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def draw_mu(mask, img):
    import random
    # mask = cv2.imread(mask_dir + item)
    # b, g, r = cv2.split(mask)
    if type(mask) == type(None):
        return mask
    #if mask.shape[2] > 1:
    #   mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    R_img, G_img, B_img = np.zeros(mask.shape, np.uint8), np.zeros(mask.shape, np.uint8), np.zeros(mask.shape, np.uint8)
    label_num = np.max(mask) + 1
    # print(label_num)
    for i in range(1, label_num):
        r = random.random()*255
        g = random.random()*255
        b = random.random()*255
        R_img[mask == i] = r
        G_img[mask == i] = g
        B_img[mask == i] = b

    check = cv2.merge([R_img, G_img, B_img])
    dst=cv2.addWeighted(img,0.7,check,0.3,0)  
    return dst

def get_bboxes_ctw(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    tags = []
    with open(gt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            gt = line.split(',')
            x1 = np.int(float(gt[0]))
            y1 = np.int(float(gt[1]))
            bbox = [np.int(float(gt[i])) for i in range(4, 32)]
            bbox = np.asarray(bbox) + ([x1, y1] * 14)
            # print(len(bbox))
            bboxes.append(bbox)
    return np.array(bboxes), tags

def get_bboxes_quad(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    tags = []
    with open(gt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            gt = line.split(',')
            if gt[-1][0] == '#':
                tags.append(False)
            else:
                tags.append(True)
            box = [int(gt[i]) for i in range(8)]
            bboxes.append(box)
    return np.array(bboxes), tags

def get_bboxes(img, gt_path, data_type='ctw'):
    if data_type == 'ctw':
        return get_bboxes_ctw(img, gt_path)
    if data_type == 'quad':
        return get_bboxes_quad(img, gt_path)

def get_kernal_bboxes(rate, img, bboxes):
    # rate = 0.3
    gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
    kernal_bboxes = shrink(bboxes, rate)
    for i in range(bboxes.shape[0]):
        cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
    # drwa = draw_mu(gt_kernal)
    return gt_kernal
    # cv2.imwrite('gt_text_'+ str(rate) + '.png', drwa)

def loader_data(img_path, gt_path, save_check_dir, data_type):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_gt_path = os.path.join(save_check_dir, img_name + '_gt_text.png')
    img = cv2.imread(img_path)
    bboxes, tags = get_bboxes(img, gt_path, data_type)
    gt_text = np.zeros(img.shape[0:2], dtype='uint8')
    training_mask = np.ones(img.shape[0:2], dtype='uint8')
    bboxes = np.reshape(bboxes, (bboxes.shape[0],int(bboxes.shape[1]/2), 2)).astype('int32')
    for i in range(bboxes.shape[0]):
        cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
    cv2.imwrite(save_gt_path, draw_mu(gt_text, img))

    for rate in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        gt_kernal = get_kernal_bboxes(rate, img, bboxes)
        drwa = draw_mu(gt_kernal, img)
        save_gt_path = os.path.join(save_check_dir, img_name + '_gt_text_' + str(rate) + '.png')
        cv2.imwrite(save_gt_path, drwa)

if __name__ == '__main__':
    img_path = '/work/ocr/detector/CTD/data/ctw1500/train/text_image/0201_A1AM03_LSVAF418XB2268959_3290006506903_20110727.jpg'
    gt_path = '/work/ocr/detector/CTD/data/ctw1500/train/text_label_curve/0201_A1AM03_LSVAF418XB2268959_3290006506903_20110727.txt'
    save_check_dir = '/work/ocr/detector/PSENet/test/my'
    data_type = 'ctw'
    loader_data(img_path, gt_path, save_check_dir, data_type)
    