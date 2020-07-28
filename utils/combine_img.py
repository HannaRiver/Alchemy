import os
import cv2
import numpy as np


def combine_two_img(img1, img2):
    image = np.concatenate([img1, img2], axis=0)
    return image

def combine2dir(dir1, dir2, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for item in os.listdir(dir1):
        if not os.path.exists(os.path.join(dir2, item)):
            continue
        img1 = cv2.imread(os.path.join(dir1, item))
        img2 = cv2.imread(os.path.join(dir2, item))
        img = combine_two_img(img1, img2)

        cv2.imwrite(os.path.join(save_dir, item), img)

if __name__ == '__main__':
    for item in ['9RegisterDate', '10IssueDate', '4address', '1PlateNo', '5UseCharacter', '8EngineNo', '7VIN', '3Owner', '2VehicleType', '6Model']:
        dir1 = os.path.join('/work/ocr/card/vehicle_license/data/test/cut', item, 'char_seg_v000_200000')
        dir2 = os.path.join('/work/ocr/card/vehicle_license/data/test/cut', item, 'char_seg_v001_140000_nms0499')
        save_dir = os.path.join('/work/ocr/card/vehicle_license/data/test/cut', item, 'combi_v000_v001')
        combine2dir(dir1, dir2, save_dir)