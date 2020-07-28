#/usr/bin/python3
#-*- coding=utf-8 -*-
'''
根据人工check的roi的label对原始的mark，xml，json进行改名（大写日期类）
'''
import os
import re
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'unity'))

from functools import reduce
from write_xml import modifyXml
from ca_rename import isfileExist
from cd2lcd import cd2lcd

def cd_rename(data_root):
    roi_dir = os.path.join(data_root, 'roi')
    mark_dir = os.path.join(data_root, 'mark')
    # mark_dir = os.path.join(data_root, 'chn')
    xml_dir = os.path.join(data_root, 'xml')
    json_dir = os.path.join(data_root, 'json')

    for roi_file in os.listdir(roi_dir):
        # print(roi_file)
        cd_value = roi_file.split('_')[-1][: -4]
        pre_name = roi_file.split(cd_value)[0]

#######################################################
        tmp1, tmp2 = cd_value.split('月')
        # if '零' == tmp2[0]:
        #     tmp2 = tmp2[1:]
        if tmp2 == '叁拾零日':
            cd_value = tmp1 + '月叁拾日'
        elif tmp2 == '壹拾零日':
            cd_value = tmp1 + '月壹拾日'
        elif tmp2 == '贰拾零日':
            cd_value = tmp1 + '月贰拾日'
        else:
            cd_value = tmp1 + '月' + tmp2     
############################################################
        lcd_value = cd2lcd(cd_value)

        assert(len(lcd_value) == 8), cd_value + lcd_value

        mark_path = os.path.join(mark_dir, pre_name + lcd_value + '.png')
        xml_path = os.path.join(xml_dir, pre_name + lcd_value + '.xml')
        json_path = os.path.join(json_dir, pre_name + lcd_value + '.png.json')

        if reduce(lambda x, y: x and y, [isfileExist(i) for i in [mark_path, xml_path, json_path]]):
            os.rename(os.path.join(roi_dir, roi_file), os.path.join(roi_dir, pre_name + cd_value + '.png'))
            os.rename(mark_path, os.path.join(mark_dir, pre_name + cd_value + '.png'))
            os.rename(json_path, os.path.join(json_dir, pre_name + cd_value + '.png.json'))
            modifyXml(xml_path, pre_name+cd_value+'.png', os.path.join(xml_dir, pre_name + cd_value + '.xml'))
            os.remove(xml_path)
        else:
            if os.path.exists(os.path.join(mark_dir, pre_name + cd_value + '.png')):
                continue
            print("ROI ERROR: ", os.path.join(roi_dir, roi_file))

if __name__ == '__main__' :
    data_root = sys.argv[1]
    cd_rename(data_root)

