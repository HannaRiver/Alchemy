#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import sys
import cv2
import random
import numpy as np
from imgaug import augmenters as iaa
from alchemy_config import cfg
sys.path.append(UTILS_DIR)
from file2list import readTxt


random_channals = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]

def add_color(image, change_channal, rgb_value):
    aug_no_colorspace = iaa.WithChannels(change_channal, iaa.Add(rgb_value))
    return aug_no_colorspace.augment_image(image)

def img_addElementwise(image_path, save_image_dir, random_times=1, is_show=False, aug_way='addElewise'):
    '''
    对于每个像素进行随机加值，value取值范围为[-30, 30]
    image_path: 图片路径;
    save_image_dir: 保存图片的根目录
    random_times: 拓增次数
    is_show: 是否看拓增图
    '''
    if is_show:
        print("aug image: %s" %(image_path))
    if not os.path.exists(save_image_dir) and not is_show:
        os.mkdir(save_image_dir)
        print("save aug image dir not exists! --> make it. path: %s" %(save_image_dir))
    image_name, image_suffix= os.path.splitext(os.path.basename(image_path))
    image = cv2.imread(image_path)
    for i in range(random_times):
        change_channal = random_channals[random.randint(0, 6)]
        change_channal_tmp = change_channal if type(change_channal) != type(1) else [change_channal]
        dropout_ratio = random.randint(0, random_times)
        aug = iaa.AddElementwise((-30, 30), True) if aug_way=='addElewise' else iaa.AddToHueAndSaturation((-20, 20), per_channel=True)
        changed_image = aug.augment_image(image)
        save_image_name = image_name + '_' + aug_way + '_' + ''.join(str(i) for i in change_channal_tmp) + '_' + str(dropout_ratio) + image_suffix
        save_image_path = os.path.join(save_image_dir, save_image_name)
        if is_show:
            img_show = np.concatenate([image, changed_image], axis=0)
            cv2.imshow("AddElementwise orig - changed", img_show)
            cv2.waitKey(0)
        else:
            cv2.imwrite(save_image_path, changed_image) 

def batch_img_addElementwise(image_dir, save_image_dir='', random_times=1, is_show=False, aug_way='addElewise'):
    '''
    批量对于每个像素进行随机加值，value取值范围为[-30, 30]
    image_dir: 图片根目录，支持txt绝对路径输入（小bug: image_dir 不要以'/'结尾）;
    save_image_dir: 如果为'', 默认保存为image_dir上一级'_addElementwise'文件夹
    random_times: 每张图片拓增次数
    is_show: 是否看拓增图
    '''
    print("========================================")
    if os.path.isfile(image_dir):
        img_list = readTxt(image_dir)
        IsFile = True
    elif os.path.isdir(image_dir):
        img_list = os.listdir(image_dir)
        IsFile = False
    else:
        pass
        print("Error: 不支持的路径输入 -> %s" %(image_dir))
    print("Deal with folder/txt file: %s[size: %s]" %(image_dir, len(img_list)))
    save_image_dir = save_image_dir if len(save_image_dir) != 0 else os.path.splitext(image_dir)[0] + '_' + aug_way
    print('save image dir: %s' %(save_image_dir))
    print("imgaug meta: %s \t aug times: %s\n" %(aug_way, random_times))
    for image_item in img_list:
        random_key = random.randint(0, 9)
        if random_key > 3:
            continue
        image_path = image_item if IsFile else os.path.join(image_dir, image_item)
        img_addElementwise(image_path, save_image_dir, random_times, is_show, aug_way)
    print("=========== %s Aug Done ==========" %(aug_way))

def img_dropout(image_path, save_image_dir, aug_type='WithChannels', random_times=1, is_show=False):
    '''
    将图像转换为hsv空间，对hsv空间随机通道dropout随机值，变化值的取值范围为[0, 0.03]
    image_path: 图片路径;
    save_image_dir: 保存图片的根目录
    random_times: 拓增次数
    is_show: 是否看拓增图
    '''
    if is_show:
        print("aug image: %s" %(image_path))
    if not os.path.exists(save_image_dir) and not is_show:
        os.mkdir(save_image_dir)
        print("save aug image dir not exists! --> make it. path: %s" %(save_image_dir))
    image_name, image_suffix= os.path.splitext(os.path.basename(image_path))
    image = cv2.imread(image_path)
    for i in range(random_times):
        change_channal = random_channals[random.randint(0, 6)]
        change_channal_tmp = change_channal if type(change_channal) != type(1) else [change_channal]
        dropout_ratio = random.random() * 0.03
        aug = iaa.WithColorspace("BGR", "HSV", children=iaa.WithChannels(change_channal, iaa.Dropout(dropout_ratio))) if aug_type=='WithChannels' else iaa.WithChannels(change_channal, iaa.Dropout(dropout_ratio))
        aug_way = '_hsvdropout_' if  aug_type=='WithChannels' else '_bgrdropout_'
        changed_image = aug.augment_image(image)
        save_image_name = image_name + aug_way + ''.join(str(i) for i in change_channal_tmp) + '_' + str(dropout_ratio) + image_suffix
        save_image_path = os.path.join(save_image_dir, save_image_name)
        if is_show:
            img_show = np.concatenate([image, changed_image], axis=0)
            print("change range: [%s, %s], dropout ratio: %s, change channal: %s\n"
                  %(0, 0.03, dropout_ratio, change_channal))
            cv2.imshow("WithColorspace-Dropout orig - changed", img_show)
            cv2.waitKey(0)
        else:
            cv2.imwrite(save_image_path, changed_image) 

def img_withchannels(image_path, save_image_dir, random_times=5, is_show=False):
    '''
    随机在BGR空间中+-随机值，变化值取值范围为[下界的1/3, 上界的1/2]
    image_path: 图片路径;
    save_image_dir: 保存图片的根目录
    random_times: 拓增次数
    is_show: 是否看拓增图
    '''
    if is_show:
        print("aug image: %s" %(image_path))
    if not os.path.exists(save_image_dir) and not is_show:
        os.mkdir(save_image_dir)
        print("save aug image dir not exists! --> make it. path: %s" %(save_image_dir))
    image_name, image_suffix= os.path.splitext(os.path.basename(image_path))
    image = cv2.imread(image_path)
    mean_value = [np.mean(image[:,:,i]) for i in [0, 1, 2]]
    bgr_lower_bounds = list(map(lambda x: int(x/3), mean_value))
    bgr_upper_bounds = list(map(lambda x: int((255 - x)/2), mean_value))
    for i in range(random_times):
        change_channal = random_channals[random.randint(0, 6)]
        change_channal_tmp = change_channal if type(change_channal) != type(1) else [change_channal]
        bgr_lower_bound = min([bgr_lower_bounds[i] for i in change_channal_tmp])
        bgr_upper_bound = min([bgr_upper_bounds[i] for i in change_channal_tmp])
        bgr_change_value = random.randint(-bgr_lower_bound, bgr_upper_bound)
        changed_image = add_color(image, change_channal, bgr_change_value)
        save_image_name = image_name + '_bgradd_' + ''.join(str(i) for i in change_channal_tmp) + '_' + str(bgr_change_value) + image_suffix
        save_image_path = os.path.join(save_image_dir, save_image_name)
        if is_show:
            img_show = np.concatenate([image, changed_image], axis=0)
            print("change range: [%s, %s], change value: %s, change channal: %s\n"
                  %(bgr_lower_bound, bgr_upper_bound, bgr_change_value, change_channal))
            cv2.imshow("WithChannels orig - changed", img_show)
            cv2.waitKey(0)
        else:
            cv2.imwrite(save_image_path, changed_image)

def img_withcolorspace(image_path, save_image_dir, random_times=5, is_show=False):
    '''
    将图像转换为hsv空间，对hsv空间随机通道+-随机值，变化值的取值范围为[-9, 9]
    image_path: 图片路径;
    save_image_dir: 保存图片的根目录
    random_times: 拓增次数
    is_show: 是否看拓增图
    '''
    if is_show:
        print("aug image: %s" %(image_path))
    if not os.path.exists(save_image_dir) and not is_show:
        os.mkdir(save_image_dir)
        print("save aug image dir not exists! --> make it. path: %s" %(save_image_dir))
    image_name, image_suffix= os.path.splitext(os.path.basename(image_path))
    image = cv2.imread(image_path)
    for i in range(random_times):
        change_channal = random_channals[random.randint(0, 6)]
        change_channal_tmp = change_channal if type(change_channal) != type(1) else [change_channal]
        hsv_change_value = random.randint(-9, 9)
        aug = iaa.WithColorspace("BGR", "HSV", children=iaa.WithChannels(change_channal, iaa.Add(hsv_change_value)))
        changed_image = aug.augment_image(image)
        save_image_name = image_name + '_hsvadd_' + ''.join(str(i) for i in change_channal_tmp) + '_' + str(hsv_change_value) + image_suffix
        save_image_path = os.path.join(save_image_dir, save_image_name)
        if is_show:
            img_show = np.concatenate([image, changed_image], axis=0)
            print("change range: [%s, %s], change value: %s, change channal: %s\n"
                  %(-9, 9, hsv_change_value, change_channal))
            cv2.imshow("WithColorspace orig - changed", img_show)
            cv2.waitKey(0)
        else:
            cv2.imwrite(save_image_path, changed_image)

def aug_type2AddColorAug(aug_type):
    function_list = {'WithChannels': img_withchannels, 'WithColorspace': img_withcolorspace}
    return [function_list[i] for i in aug_type]

def batch_add_color(image_dir, save_image_dir='', aug_type=['WithColorspace', 'WithChannels'], random_times=5, is_show=False):
    '''
    WithColorspace: 对HSV中变换; WithChannels: 对BGR变换
    批量随机在BGR空间中+-随机值，变化值取值范围为[下界的1/3, 上界的1/2]
    image_dir: 图片根目录，支持txt绝对路径输入（小bug: image_dir 不要以'/'结尾）;
    save_image_dir: 如果为'', 默认保存为image_dir上一级'_WithChannels'文件夹
    random_times: 每张图片拓增次数
    is_show: 是否看拓增图
    '''
    print("========================================")
    if os.path.isfile(image_dir):
        img_list = readTxt(image_dir)
        IsFile = True
    elif os.path.isdir(image_dir):
        img_list = os.listdir(image_dir)
        IsFile = False
    else:
        pass
        print("Error: 不支持的路径输入 -> %s" %(image_dir))
    print("Deal with folder/txt file: %s[size: %s]" %(image_dir, len(img_list)))
    for i, func in enumerate(aug_type2AddColorAug(aug_type)):
        if len(save_image_dir) == 0:
            save_image_dir_new = os.path.splitext(image_dir)[0] + '_' + aug_type[i]  if IsFile else image_dir + '_' + aug_type[i]
        else:
            save_image_dir_new = os.path.join(save_image_dir, aug_type[i])
        print("save image dir: %s" %(save_image_dir_new))
        print("imgaug meta: %s-Add \t aug times: %s\n" %(aug_type[i],random_times))
        for item in img_list:
            random_key = random.randint(0, 9)
            if random_key > 3:
                continue
            image_path = os.path.join(image_dir, item) if not IsFile else item
            func(image_path, save_image_dir_new, random_times, is_show)
    print("=========== WithChannels Add Aug Done ==========")

def batch_dropout_color(image_dir, save_image_dir='', aug_types=['WithColorspace', 'WithChannels'], random_times=5, is_show=False):
    '''
    WithColorspace: 对HSV中变换; WithChannels: 对BGR变换
    批量随机在BGR/HSV空间中随机失活，dropout ratio: [0, 0.03]
    image_dir: 图片根目录，支持txt绝对路径输入（小bug: image_dir 不要以'/'结尾）;
    save_image_dir: 如果为'', 默认保存为image_dir上一级'_WithChannels_dropout'文件夹
    random_times: 每张图片拓增次数
    is_show: 是否看拓增图
    '''
    print("========================================")
    if os.path.isfile(image_dir):
        img_list = readTxt(image_dir)
        IsFile = True
    elif os.path.isdir(image_dir):
        img_list = os.listdir(image_dir)
        IsFile = False
    else:
        pass
        print("Error: 不支持的路径输入 -> %s" %(image_dir))
    print("Deal with folder/txt file: %s[size: %s]" %(image_dir, len(img_list)))
    for aug_type in aug_types:
        if len(save_image_dir) == 0:
            save_image_dir_new = os.path.splitext(image_dir)[0] + '_' + aug_type  if IsFile else image_dir + '_' + aug_type + '_dropout'
        else:
            save_image_dir_new = os.path.join(save_image_dir, aug_type + '_dropout')
        print('save image dir: %s' %(save_image_dir_new))
        print("imgaug meta: %s-Dropout \t aug times: %s\n" %(aug_type,random_times))
        for item in img_list:
            random_key = random.randint(0, 9)
            if random_key > 3:
                continue
            image_path = os.path.join(image_dir, item) if not IsFile else item
            img_dropout(image_path, save_image_dir_new, aug_type, random_times, is_show)
    print("=========== WithChannels Dropout Aug Done ==========")

if __name__ == '__main__':
    image_dir = sys.argv[1]
    image_dir = image_dir if image_dir[-1] != '/' else image_dir[: -1]
    if len(sys.argv) == 2:
        batch_add_color(image_dir, save_image_dir='', aug_type=['WithColorspace', 'WithChannels'], random_times=1, is_show=False)
        batch_dropout_color(image_dir, save_image_dir='', aug_types=['WithColorspace', 'WithChannels'], random_times=1, is_show=False)
        batch_img_addElementwise(image_dir, save_image_dir='', random_times=1, is_show=False, aug_way='addElewise')
        batch_img_addElementwise(image_dir, save_image_dir='', random_times=1, is_show=False, aug_way='AddToHueAndSaturation')
    else:
        aug_type = [int(i) for i in sys.argv[2: ]]
        func_list = [batch_add_color, batch_dropout_color, batch_img_addElementwise]
        assert(len(aug_type) <= len(func_list)), '支持0, 1, 2 对应: batch_add_color, batch_dropout_color, batch_img_addElementwise'
        for i in aug_type:
            func_list[i](image_dir, random_times=1)









