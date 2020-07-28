#/usr/bin/python3
#-*- encoding=utf-8 -*-
'''
根据人工check的roi的label对原始的mark，xml，json进行改名（大写金额类）
'''
import os
import re
import sys
import glob
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'unity'))

from functools import reduce
from ca2lca import ca2lca
from write_xml import modifyXml



def isfileExist(file_path):
    if not os.path.exists(file_path):
        # print("Warning :: file [%s] not exists!" %(file_path))
        return False
    return True

def ca_rename(data_root):
    roi_dir = os.path.join(data_root, 'roi')
    mark_dir = os.path.join(data_root, 'mark')
    xml_dir = os.path.join(data_root, 'xml')
    json_dir = os.path.join(data_root, 'json')

    for roi_file in os.listdir(roi_dir):
        # print(roi_file)
        ca_value = roi_file.split('_')[-1][: -4]
        pre_name = roi_file.split(ca_value)[0]
        
        lca_value = ca2lca(ca_value)

        mark_path = os.path.join(mark_dir, pre_name + lca_value + '.png')
        xml_path = os.path.join(xml_dir, pre_name + lca_value + '.xml')
        json_path = os.path.join(json_dir, pre_name + lca_value + '.png.json')

        if reduce(lambda x, y: x and y, [isfileExist(i) for i in [mark_path, xml_path, json_path]]):
            os.rename(mark_path, os.path.join(mark_dir, pre_name + ca_value + '.png'))
            os.rename(json_path, os.path.join(json_dir, pre_name + ca_value + '.png.json'))
            modifyXml(xml_path, pre_name+ca_value+'.png', os.path.join(xml_dir, pre_name + ca_value + '.xml'))
            os.remove(xml_path)
        else:
            if os.path.exists(os.path.join(mark_dir, pre_name + ca_value + '.png')):
                continue
            print(os.path.join(roi_dir, roi_file))


def tmp_function(checked_dir, data_dir, save_dir):
    '''
    checked_dir: 人工确认正确且修正标签的图像目录
    data_dir: 原始图片的目录
    save_dir: 修改原始图片名称的保存目录
    '''
    for roi_file in os.listdir(checked_dir):
        # print(roi_file)
        ca_value = roi_file.split('_')[-1][: -4]
        pre_name = roi_file.split(ca_value)[0]

        lca_value = ca2lca(ca_value)
        # lca_value = ca_value

        img_path = os.path.join(data_dir, pre_name + lca_value + '.png')
        save_path = os.path.join(save_dir, pre_name + ca_value + '.png')
        if isfileExist(img_path):
            os.rename(img_path, save_path)
        else:
            if not isfileExist(save_path):
                # img_path = glob.glob(os.path.join(data_dir, pre_name + '*'))[0]
                # os.rename(img_path, save_path)
                print(roi_file)





if __name__ == '__main__' :
    
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item'
    save_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/CA'
    item_list = ['CA', 'Ck', 'DR']
    name_list = ['大写金额', '支票大写金额', '进账单大写金额']
    # date_list = ['20180731', '20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']
    # item_list = ['CA']
    date_list = ['20180920']# , '20180913', '20180914', '20180917']

    for date in date_list:
        for i, item in enumerate(item_list):
            # data_root = os.path.join(data_dir, item, date)
            # print("---> Deal with %s" %(data_root))
            # ca_rename(data_root)
            checked_dir = os.path.join(data_root, 'CA', item, date, 'ssd_roi')
            data_dir = os.path.join(data_root, name_list[i], date)
            save_dir = os.path.join(save_root, item, date, 'img')

            print("---> Deal with %s" %(data_dir))
            if not os.path.exists(checked_dir):
                print("NEED LABELD!!!")
                continue

            if not os.path.exists(save_dir):
                os.mkdir(os.path.join(save_root, item, date))
                os.mkdir(save_dir)
            tmp_function(checked_dir, data_dir, save_dir)
    print("Done!")
