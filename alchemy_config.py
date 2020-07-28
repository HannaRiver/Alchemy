#/usr/bin/python
#-*- encoding=utf-8 -*-
"""Alchemy config system.
This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See *.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

#
# MISC
#

# Root directory of alchemy
__C.ROOT_DIR = osp.abspath(osp.dirname(__file__))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models'))

# utils directory
__C.UTILS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'utils'))

# imgaug directory
__C.IMGAUG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'imgaug'))

# crcb directory

__C.CRCB_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'ocr', 'crcb'))

#
# Caffe root
#

__C.CAFFE_ROOT = edict()

# classfify caffe
__C.CAFFE_ROOT.CLASSIFY = '/home/hena/caffe-ssd/caffe'

# lstm caffe root 
__C.CAFFE_ROOT.LSTM = '/home/em/software/caffe-ocr_combine'

# ssd caffe root
__C.CAFFE_ROOT.SSD = '/home/em/software/ssd_caffe'

#
# crcb misc(data)
#
__C.CRCB_DATA = edict()

# need2deal dir
__C.CRCB_DATA.SPY_ZIP_DIR = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/spy/pictures'
__C.CRCB_DATA.DOWNLOAD_DIR_PRINT = '/work/ocr/handwriting/data/org_print/download'
__C.CRCB_DATA.CHECKDATA_DIR_PRINT = '/work/ocr/handwriting/data/org_print/checkdata'
__C.CRCB_DATA.TRIAN_REC_DIR_PRINT = '/data1/ocr/data/rec'
__C.CRCB_DATA.SHARE_SERVER = '192.168.20.252'
__C.CRCB_DATA.SHARE_FOLDER = 'cls_item'
__C.CRCB_DATA.SHARE_ISHD = True #下载的数据是否是手写体
# 数据部标好定位的数据地址
__C.CRCB_DATA.DETLABELED_DIR = '/data1/ocr/data/det'
# 需要处理定位数据的日期批次
__C.CRCB_DATA.DETLABELED_KEYDATE = '20181024'
__C.CRCB_DATA.KEYDATE = '20181024'
__C.CRCB_DATA.RECITEM = 'CA'
__C.CRCB_DATA.CHECKSTATE = 'check_right'
# 需要end2end识别的日期批次
__C.CRCB_DATA.DET2REC_KEYDATE = '20180914,20181127' # [20180914]
__C.CRCB_DATA.DET2REC_RECITEM = 'CA'

#
# crcb wordslib
#

__C.CRCB_WORDSLIB = edict()

__C.CRCB_WORDSLIB.IMG_TXT = 'path/to/img'
__C.CRCB_WORDSLIB.LABELED_TXT = 'path/to/save/labeled'
# shred_chn
__C.CRCB_WORDSLIB.SHRED_CHN = os.path.join(__C.CRCB_DIR, 'wordslib', 'shred_chn.pkl')

# shred ca
__C.CRCB_WORDSLIB.CA_LIST = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
                             '拾', '元', '佰', '仟', '万', '亿', '角', '分', '整', '正',
                             '圆', ' ']
__C.CRCB_WORDSLIB.CA_NUMMAX = 20

# shred cd
__C.CRCB_WORDSLIB.CD_LIST = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
                             '拾', '年', '月', '日', ' ']
__C.CRCB_WORDSLIB.CD_NUMMAX = 12

# shred la
__C.CRCB_WORDSLIB.LA_LIST = ["￥","0","1","2","3","4","5","6","7","8","9", " "]

#
# crcb cfg path
#
__C.CRCB_CFG_NAME = edict()

# item2cfg
__C.CRCB_CFG_NAME.CA_PRINT = 'ca_print_cfg.yml'
__C.CRCB_CFG_NAME.LA_PRINT = 'la_print_cfg.yml'
__C.CRCB_CFG_NAME.CD_PRINT = 'cd_print_cfg.yml'
__C.CRCB_CFG_NAME.LD_PRINT = 'ld_print_cfg.yml'
__C.CRCB_CFG_NAME.CHN_PRINT = 'chn_print_cfg.yml'
__C.CRCB_CFG_NAME.NUM_PRINT = 'num_print_cfg.yml'

#
# crcb model path
#

__C.CRCB_MODEL = edict()

# crcb model dir
__C.CRCB_MODEL.DIR = osp.join(__C.MODELS_DIR, 'ocr', 'crcb')
# shred CA model
__C.CRCB_MODEL.SHRED_DET_LABELMAP = osp.join(__C.CRCB_MODEL.DIR, 'det', 'ca', 'labelmap.prototxt')
__C.CRCB_MODEL.SHRED_DET_MODELDEF = osp.join(__C.CRCB_MODEL.DIR, 'det', 'ca', 'CapitalMoneyRecogSSD.prototxt')
__C.CRCB_MODEL.SHRED_DET_MODELWEIGHT = osp.join(__C.CRCB_MODEL.DIR, 'det', 'ca', 'CapitalMoneyRecogSSD.caffemodel')
__C.CRCB_MODEL.SHRED_DET_MEANVALUE = [185, 185, 185]
__C.CRCB_MODEL.SHRED_DET_SCALE = 1
__C.CRCB_MODEL.SHRED_DET_RESIZETYPE = ''
__C.CRCB_MODEL.SHRED_DET_RESIZESIZE = [160, 480]

__C.CRCB_MODEL.SHRED_REC_MODELDEF = osp.join(__C.CRCB_MODEL.DIR, 'rec', 'ca', 'CapitalMoneyRecogLSTM.prototxt')
__C.CRCB_MODEL.SHRED_REC_MODELWEIGHT = osp.join(__C.CRCB_MODEL.DIR, 'rec', 'ca', 'CapitalMoneyRecogLSTM.caffemodel')
__C.CRCB_MODEL.SHRED_REC_MEANVALUE = [150, 139, 138]
__C.CRCB_MODEL.SHRED_REC_SCALE = 1
__C.CRCB_MODEL.SHRED_REC_TIMESTEP = 60
__C.CRCB_MODEL.SHRED_REC_RESIZETYPE = 'undeform_resize'
__C.CRCB_MODEL.SHRED_REC_RESIZESIZE = [48, 240]




def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value