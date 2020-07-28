#/usr/bin/python2
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.append(os.path.join(tool_root, 'det'))
import cv2

from plate_locate_ssd import main as save_ssd_result


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def get_fake_vai_cardata(car_img, plate_img, save_path):
    resize_plate_img = cv2.resize(plate_img, (80, 50), interpolation=cv2.INTER_CUBIC)
    roi = car_img[160: 210, 50: 130]
    # print(roi.shape)
    # print(resize_plate_img.shape)
    car_img[160: 210, 50: 130] = resize_plate_img

    cv2.imwrite(save_path, car_img)
    # cv2.namedWindow("ssd result", cv2.WINDOW_NORMAL)
    # cv2.imshow("ssd result", car_img)
    # cv2.waitKey(0)

def get_single_test(car_path, plate_path, save_path):
    car_img = cv2.imread(car_path)
    plate_img = cv2.imread(plate_path)
    get_fake_vai_cardata(car_img, plate_img, save_path)

def bath_get_test(car_path, plate_dir, save_dir):
    car_img = cv2.imread(car_path)

    for item in os.listdir(plate_dir):
        plate_path = os.path.join(plate_dir, item)
        save_path = os.path.join(save_dir, item)
        plate_img = cv2.imread(plate_path)
        get_fake_vai_cardata(car_img, plate_img, save_path)

def batch_dir_test(plate_dir):        
    save_dir = os.path.join(plate_dir, '..', 'fake_data')
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    
    # bath_get_test(car_path, plate_dir, save_dir)
    # print("=============== fack precess done ========================")
    save_ssd_result(save_dir)

if __name__ == '__main__':
    car_path = '/home/hena/图片/test_car.png'
    # plate_path = '/work/hena/ocr/data/CLPR/VAI/double_yellow/chepai/0100_test/double/img/20180120_chepai_prefix_wrong_17.jpg'
    # img_name = os.path.basename(plate_path)
    # save_path = os.path.join('/home/hena/图片/tmp_save', img_name)
    # get_single_test(car_path, plate_path, save_path)

    # plate_dir = '/work/hena/ocr/data/CLPR/VAI/double_yellow/chepai/0100_test/double/img'
    # batch_dir_test(plate_dir)

    plate_list = readTxt('/work/hena/ocr/data/CLPR/VAI/double_yellow/chepai/double_plate.txt')
    for plate_dir in plate_list:
        batch_dir_test(plate_dir)
        print("==>> " + plate_dir + "--> Done!")

    

