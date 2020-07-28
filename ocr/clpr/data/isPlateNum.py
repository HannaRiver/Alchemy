#/usr/bin/python3
#-*- encoding=utf-8 -*-


def isPlateNum(plate_num):
    '''
    初步判断是否满足车牌的命名规范
    车牌位数为7-8位， 第一位一定不为数字
    '''
    if len(str(plate_num)) < 7 or len(str(plate_num)) > 8:
        return False
    if str(plate_num)[0].isdigit():
        return False
    return True