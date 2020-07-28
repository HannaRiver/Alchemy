#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from file2list import get_label4name


def batch_download_idcard_json(idcard_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    idcard_list = os.listdir(idcard_dir)
    for idcard_name in idcard_list:
        item = get_label4name(get_label4name(idcard_name, 1, '_'), 0, '.')
        os.system('aria2c -d %s/ http://192.168.40.126:3000/api/idcards/%s' % (save_dir, item))


        
if __name__ == '__main__':
    txt_path = '/work/hena/ocr/data/Card/idcard/output'
    save_dir = '/work/hena/ocr/data/Card/idcard/json'
    batch_download_idcard_json(txt_path, save_dir)