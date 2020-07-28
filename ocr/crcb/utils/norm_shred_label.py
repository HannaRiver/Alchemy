#/usr/bin/python3
#-*- coding=utf-8 -*-
from lca2ca import cncurrency
# from ca2lca import ca2lca

def normShredLabel(label, rec_item, rec):
    '''
    对银行标签进行归一化转换
    '''
    if rec_item == 'CA':
        ca_label = cncurrency(label)
        norm_label = ca_label.replace('整', '') if '整' in ca_label else ca_label
        norm_rec = rec.replace('正', '') if '正' in rec else rec
        norm_rec = norm_rec.replace('整', '') if '整' in norm_rec else norm_rec
        norm_rec = norm_rec.replace('圆', '元') if '圆' in norm_rec else norm_rec
        norm = rec if norm_rec == norm_label else ca_label
        return norm
    elif rec_item == 'LA':
        norm = label if '￥' not in rec else '￥' + label
        return norm
    else:
        pass
    return label