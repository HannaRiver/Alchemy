"""
两张图片进行拼接及比较，包含需要分析的若干模块
"""
import os
import logging
import math
import cv2
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# import seaborn as sns


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

class ImgComparison():
    pass

def img_color(img, bgr=True, heat=False):
    color_translate = [(0, 0, 0), (1, 159, 246), (0, 236, 236), (1, 216, 0), (1, 144, 0),
                       (255, 255, 0), (231, 192, 0), (255, 144, 0),
                       (254, 0, 0), (214, 0, 0), (192, 0, 0),
                       (255, 0, 240), (149, 0, 180), (174, 144, 240)]  # RGB

    if heat:
        color_translate = [(0, 0, 0), (0, 0, 255), (0, 84, 255), (0, 168, 255), (0, 255, 255),
                           (0, 255, 168), (0, 255, 84), (0, 255, 0),
                           (84, 255, 0), (168, 255, 0), (255, 255, 0),
                           (255, 168, 0), (255, 84, 0), (255, 0, 0)]  # RGB
    if not bgr:
        #  color_translate = [i for i, j in enumerate(color_translate)]
        color_translate = [range(len(color_translate))]

    cl = len(color_translate)

    mm = np.array(color_translate)
    ind = (img / 5).astype('int')
    ind -= 1
    ind = np.clip(ind, a_min=0, a_max=cl - 1)

    res = mm[ind]
    return res

def remove_outside(src):
    h, w = src.shape[: 2]
    for i in range(h):
        for j in range(w):
            if (i-h/2)**2 + (j-w/2)**2 > (h/2)**2:
                src[i, j] = 0.0
    return src

def img_stitching(srcs, colors=[[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]):
    """Stitch two image.

    Args:
        src: Image 1, suggest use the original image.
        dst: Image 2, stitched image.
        colors: draw the border color.

    Returns:
            red      green
        |--------||--------|
        | img 1  || img 2  |
        |--------||--------|

    """
    # if src.shape != dst.shape:
    #     logging.error("Two Image Shape are differ.")
    #     return
    src = srcs[0]
    img_num = len(srcs)
    if len(colors) < img_num: colors=[[0, 0, 255], [0, 255, 0], [255, 0, 0]]*math.ceil(img_num/3)
    height, width = src.shape[: 2]
    src = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=colors[0])
    for i in range(1, len(srcs)):
        dst = cv2.copyMakeBorder(srcs[i], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=colors[i])
        src = np.concatenate([src, dst], axis=1)
    
    return src

def test_img_stitching(src_dir, src_gary_dir, dst_dir, dst_gary_dir, save_dir, _DEBUG=False):
    if _DEBUG:
        logging.info('''\n
        =====================================================\n
        Test Img Stiching -> save dir => %s\n
        |--------||--------||--------||---------||----------|\n
        |   gt   ||   pre  || pre-gt || miss gt || overflow |\n
        |--------||--------||--------||---------||----------|\n
        =====================================================''' %(save_dir))
    if not os.path.exists(src_dir):
        logging.error("The first image path not exists! -> ", src_dir)
        return
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    cloud_nums, miss_nums, error_nums = [], [], []
    for img_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        
        if not os.path.exists(dst_path):
            logging.error("The sec image path not exists! -> ", dst_path)
            continue
        src = cv2.imread(src_path)
        dst = remove_outside(cv2.imread(dst_path))

        src_gary = cv2.imread(os.path.join(src_gary_dir, img_name), 0)
        dst_gary = cv2.imread(os.path.join(dst_gary_dir, img_name), 0)
        diff = remove_outside(dst_gary) - remove_outside(src_gary)
        diff = img_color(diff)

        _, src_bi = cv2.threshold(src_gary, 65, 255, cv2.THRESH_BINARY)
        _, dst_bi = cv2.threshold(dst_gary, 65, 255, cv2.THRESH_BINARY)
        mask1 = dst_bi - src_bi
        mask2 = src_bi - dst_bi
        diff2 = cv2.bitwise_and(src, src, mask=mask2)
        diff4 = cv2.bitwise_and(dst, dst, mask=mask1)

        shape = mask1.shape
        all_num = shape[0]*shape[1]
        # diff2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cloud_num = (src_bi.reshape(-1, ) == 255).sum()  
        # miss_num = (mask.reshape(-1, ) == -255).sum() 
        # 本身云数量 漏预测的数量 预测数量 误报数量
        info_nums = list(map(lambda x: (x.reshape(-1, ) == 255).sum(), [src_bi, mask2, dst_bi, mask1]))
        font=cv2.FONT_HERSHEY_SIMPLEX
        info_img = np.ones(src.shape, np.uint8)*255
        cloud_num, miss_num, error_num = round(info_nums[0]/all_num, 2), round(info_nums[1]/info_nums[0], 2), round(info_nums[3]/info_nums[2], 1)
        # cv2.putText(info_img, 'p(cloud): '+str(cloud_num),(0,30), font, 1, [0, 255, 0], 1)
        # cv2.putText(info_img, 'p(miss): '+str(miss_num),(0,60), font, 1, [0, 255, 0], 1)
        # cv2.putText(info_img, 'p(error): '+str(error_num),(0,90), font, 1, [0, 255, 0], 1)
        cv2.putText(info_img, ''+str(cloud_num),(0,30), font, 1, [0, 255, 0], 1)
        cv2.putText(info_img, ''+str(miss_num),(0,60), font, 1, [0, 255, 0], 1)
        cv2.putText(info_img, ''+str(error_num),(0,90), font, 1, [0, 255, 0], 1)
        cloud_nums.append(cloud_num)
        miss_nums.append(miss_num)
        error_nums.append(error_num)

        stitiched = img_stitching([src, dst, diff, diff2, diff4, info_img])
        cv2.imwrite(os.path.join(save_dir, img_name), stitiched)
    convert_cloud2miss(cloud_nums, miss_nums, error_nums)

def convert_cloud2miss(cloud_nums, miss_nums, error_nums):
    max_data = np.array([cloud_nums, miss_nums, error_nums]).max()
    # bins = np.linspace(0, max_data, max_data+1)

    x, y = np.array(cloud_nums), np.array(miss_nums)
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    fig, ax = plt.subplots()
    ax.plot(x, y_est, '-')
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.scatter(x, y, c="g", alpha=0.5)
    plt.hist(cloud_nums, 100, normed=True, color="#FF0000", alpha=.9)
    plt.hist(miss_nums, 100, normed=True, color="#C1F320", alpha=.5)
    plt.show()
    

def main():
    img_root = '/work/project/temp/haikou_cloud_920km_focal_loss_soft_200_test'
    src_dir = os.path.join(img_root, 'gt_color')
    src_gary = os.path.join(img_root, 'gt')
    dst_dir = os.path.join(img_root, 'pred_color')
    dst_gary = os.path.join(img_root, 'pred')
    save_dir = os.path.join(img_root, 'stitiched')
    test_img_stitching(src_dir, src_gary, dst_dir, dst_gary, save_dir, True)


if __name__ == '__main__':
    main()
    

    