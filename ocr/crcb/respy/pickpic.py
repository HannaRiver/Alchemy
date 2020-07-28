#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import sys
import cv2
import shutil
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

############################################## Analytical data ##########################################################
def crcb2dict(txtpath):
    img2savename = {}
    with open(txtpath, 'r') as f:
         for line in f.readlines():
             imgname, item, lable = line.strip().split(',')
             img2savename[imgname] = item + '_' + lable
    return img2savename

def crcb2dict_item(txtpath, item_label_list=[]):
    img2savename = {}
    with open(txtpath, 'r') as f:
         for line in f.readlines():
             imgname, item, lable = line.strip().split(',')
             # imgname, lable, item  = line.strip().split(',')
             img2savename[imgname] = item + '_' + lable

             if item not in item_label_list:
                 item_label_list.append(item)
    return img2savename, item_label_list

def batch_refilename(path, label_dir, img2savename):
    # path = "/work/hena/ocr/data/crcb/spy/0724/pictures20180724/"
    # label_dir = '/work/hena/ocr/data/crcb/label/0730_true/'
    cont = 0
    for item in os.listdir(path):
        if item[: 5] != 'dbvis':
            continue
        tmp_id = 0
        if item not in img2savename:
            logging.info(item + " not in img2savename")
            continue
        item_label, label = img2savename[item].split('_')
        # 没有label的情况
        if label == '' or label == ' ':
            continue

        cont += 1

        nonchn = ['金额', '账号', '日期', '号码', '密码', '密押', '附言', '第一背书印章名称', '最后收款人印章名称', '托收凭证收款人户名']
        if item_label[-2 :] in nonchn or label.isdigit() or label == 'ACT' or label == '' or item_label in nonchn:
            continue        
        save_name = img2savename[item] + '.png'
        while os.path.exists(path + str(tmp_id) + '_' + save_name):
            tmp_id += 1
        # savename = id_name + '_' + item_name + '_' + cncurrency(label[: -4])
        img = cv2.imread(path +item)
        cv2.imwrite(label_dir + str(tmp_id) + '_' + save_name, img)
        os.rename(path + item, path + str(tmp_id) + '_' + save_name)
    
    logging.info("Valid inforamtion: %s" %(cont))

def batch_refilename_get_item(path, label_dir, img2savename, folder, item_label_list=[]):
    logging.info("Deal with the folder: %s" %(path))

    for item in os.listdir(path):
        if item[: 5] != 'dbvis':
            logging.info("Not a org file: %s" %(item))
            continue

        img = cv2.imread(os.path.join(path, item))
        
        tmp_id = 0

        # 保存没有任何标注的数据
        if item not in img2savename:
            logging.info(item + " not in img2savename")
            save_dir = os.path.join(label_dir, 'nolabel')
            # cv2.imwrite(save_path, img)
            shutil.copy(os.path.join(path, item), os.path.join(save_dir, item))
            continue

        item_label, label = img2savename[item].split('_')
        save_name = img2savename[item] + '.png'

        # 保存的文件夹路径
        save_dir = os.path.join(label_dir, item_label)
        if not os.path.exists(save_dir):
            logging.info("folder %s not exists --> maked it" %(save_dir))
            os.mkdir(save_dir)
        
        # 保存有表向没标签的数据
        if (label == '' or label == ' ') and False:
            save_dir = os.path.join(save_dir, 'need_label')

            if not os.path.exists(save_dir):
                # logging.info("folder %s not exists --> maked it" %(save_dir))
                os.mkdir(save_dir)
            
            while os.path.exists(os.path.join(save_dir, str(tmp_id)+'_'+save_name)):
                tmp_id += 1

            cv2.imwrite(os.path.join(save_dir, str(tmp_id)+'_'+save_name), img)
            continue
        

        # 保存文件按日期分
        save_dir = os.path.join(save_dir, folder)
        if not os.path.exists(save_dir):
            # logging.info("folder %s not exists --> maked it" %(save_dir))
            os.mkdir(save_dir)

        if item_label not in item_label_list:
            item_label_list.append(item_label)
        
        
        while os.path.exists(os.path.join(save_dir, str(tmp_id)+'_'+save_name)):
            tmp_id += 1
        
        # 保存图片
        cv2.imwrite(os.path.join(save_dir, str(tmp_id)+'_'+save_name), img)
    
    return item_label_list

def main(date):
    '''
    按表项提取 crcb shred 数据
    '''
    # folderlist = ['20180828', '20180827', '20180816', '20180814', '20180813', '20180808', '20180803', '20180802', '20180731']
    folderlist = [date]
    # , '20180727']
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/spy/'
    save_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item'
    # save_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/20180925'

    item_label_list = []

    if not os.path.exists(save_root):
        os.mkdir(save_root)
        logging.info("save cls item data path %s not exists --> maked it." %(save_root))
    
    logging.info("==================== Begin Pick Picture ================")

    for folder in folderlist:
        path = os.path.join(data_root, folder, 'pictures')
        labelmap_path = os.path.join(data_root, folder, 'pictures.txt')

        img2savename, item_label_list = crcb2dict_item(labelmap_path, item_label_list)

        item_label_list = batch_refilename_get_item(path, save_root, img2savename, folder, item_label_list)
    
    logging.info("crcb shred has a total of %s class: %s" %(len(item_label_list), str(' ').join(str(i) for i in item_label_list)))
    logging.info("==================== Pick Picture Done ====================")
    # logging.info("==================== Saving Item Info ===================")

    # import pickle

    # fw = open(os.path.join(save_root, 'item_label_list.pkl'), 'wb')
    # pickle.dump(item_label_list, fw)
    # fw.close()

    # logging.info("==================== Saved Item Info ===================")







def get_chn_rename_img():
    path = "/work/hena/ocr/data/handwriting/bill/CRCB/crcb_shred/spy/pictures/"
    label_dir = '/work/hena/ocr/data/handwriting/bill/CRCB/crcb_shred/label/0726_2/'
    labelmap_path = '/work/hena/ocr/data/handwriting/bill/CRCB/crcb_shred/spy/pictures.txt'

    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

        logging.info("path %s not exists --> maked it." %(label_dir))

    logging.info("==================== Begin Pick Picture ================")
    logging.info("---------> get img2savename dict")

    img2savename = crcb2dict(labelmap_path)

    batch_refilename(path, label_dir, img2savename)

############################################### Caffe Model #########################################################


if __name__ == '__main__':
    # get_chn_rename_img() # 保存比较有价值的中文字段
    date = sys.argv[1]
    main(date)
    