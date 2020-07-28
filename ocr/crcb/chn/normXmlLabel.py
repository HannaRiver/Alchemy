#/usr/bin/python3
#-*- coding=utf-8 -*-
'''
根据人工check的roi的label对原始的mark，xml，json进行改名（大写日期类）
'''
import os
import re
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))

from write_xml import modifyXml, modifyXmlName

def chn_reLabelName(data_root, new_labelname):
    for item in os.listdir(data_root):
        xml_path = os.path.join(data_root, item)
        modifyXmlName(xml_path, new_labelname, xml_path)

if __name__ == '__main__' :
    data_root = sys.argv[1]
    new_labelname = sys.argv[2]
    chn_reLabelName(data_root, new_labelname)

