#/usr/bin/python3
#-*- encoding=utf-8 -*-


def strQ2B(ustring):
    '''
    全角转换为半角(2/3版本不兼容)
    '''
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def MyIsdigit(astr):
    '''
    1.111 是小数    1,111 是小数
    '''
    new_str = ""
    for i in astr:
        if i in [',', '.', '-']:
            continue
        new_str += i
    return new_str.isdigit()

def rmPunctuation(astr):
    '''
    去除标点符号
    '''
    pointlist = ['.', ',','-', '、', ':', '*', '(', ')', '”', '“']
    new_str = ""
    astr = strQ2B(astr)
    for i in astr:
        if i in pointlist:
            continue
        new_str += i
    return new_str

def rmParenthesesItem(astr):
    '''
    将（）中的内容，去掉;['.', ',','-', '、', '*']前面的内容去掉 只针对中文
    '''
    rstr = ''
    pointlist = ['.', ',','-', '、', '*']
    astr = strQ2B(astr)
    if rmPunctuation(astr).isdigit():
        return astr
    for i, item in enumerate(astr):
        if item in pointlist and i < 5:
            rstr = ''
            continue
        rstr += item
    return rstr

def del_chn(text):
    # "去除字符串中的中文 python2"
    return ''.join([i if ord(i) < 128 else '' for i in text])