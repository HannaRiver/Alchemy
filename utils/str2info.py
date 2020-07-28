#/usr/bin/python3
#-*- encoding=utf-8 -*-


def get_label4name(namestr, idx=0, split_key='_'):
    """
    通过解析文件名获取图像标签
    idx: 需要的内容在第几个项
    split_key: 分割字符串的key
    """
    if split_key not in namestr:
        return False
    return namestr.split(split_key)[idx]