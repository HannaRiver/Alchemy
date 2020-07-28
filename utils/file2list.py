#/usr/bin/python3
#-*- coding=utf-8 -*-


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def get_label4name(namestr, idx=0, split_key='_'):
    """
    通过解析文件名获取图像标签
    idx: 需要的内容在第几个项
    split_key: 分割字符串的key
    """
    if split_key not in namestr or type(namestr) != type('123') or len(namestr.split(split_key)) < idx+1:
        return False
    return namestr.split(split_key)[idx]

def txt2info(txtpath, idx, split_key):
    """
    txt 中每一行只有特殊字段是需要的不是所有的情况
    """
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            info = get_label4name(line.strip(), idx, split_key)
            if info:
                filelist.append(info)
    return filelist


def read_plate_vir_csv(txtpath):
    """
    处理陈伟给的csv文件，第一列为图片地址
    return -> 图片地址list
    """
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            need_info = get_label4name(line.strip(), 0, ',')
            if need_info:
                filelist.append(need_info)           
    return filelist