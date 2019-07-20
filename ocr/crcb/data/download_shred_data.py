#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import shutil
import sys
import argparse
from alchemy_config import cfg, cfg_from_file


def download_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default=cfg.CRCB_DATA.SHARE_SERVER, help='需要下载数据的ip地址')
    parser.add_argument('--share', default=cfg.CRCB_DATA.SHARE_FOLDER, help='需要下载数据的共享目录名')
    parser.add_argument('--cls_type', default=cfg.CRCB_DATA.RECITEM, help='需要下载的大类，目前支持[CA:大写金额类, LA:小写金额类, LD: 小写日期, CD: 大写日期, Num: 数字类, Chn: 中文类, Other: 其他类]')
    parser.add_argument('--is_hd', default=cfg.CRCB_DATA.SHARE_ISHD, help='是否是手写体', type=bool)
    parser.add_argument('--save_dir', default=cfg.CRCB_DATA.DOWNLOAD_DIR_PRINT, help='下载数据保存地址')
    parser.add_argument('--date', default=cfg.CRCB_DATA.KEYDATE, help='需要下载的日期key')
    parser.add_argument('--cfg_path', default='')
    return parser.parse_args()

clsType2Folder ={
    'CA': ['出票大写金额', '出票金额', '大写金额', '进账单大写金额', '实际结算金额大写', '托收凭证大写金额', '支票大写金额'],
    'LA': ['出票小写金额', '实际结算金额小写', '托收凭证小写金额', '小写金额', '支票小写金额', '进账单小写金额'],
    'CD': ['票据到期日', '凭证日期', '支票签发日期'],
    'LD': ['进账单日期', '凭证日期'],
    'Num': ['付款人行号', '付款人账号', '密押', '凭证号码', '收款人账号', '托收凭证收款人账号', '邮件编号', '支付密码', '支票付款人账号', '进账单付款人账号', '进账单收款人账号'],
    'Chn': ['付款行全称', '付款人户名', '进账单付款人户名', '进账单收款人户名', '进账单收款人户名', '收款人地址', '收款人户名', '托收凭证收款人户名', '支票收款人户名'],
    'Other': ['多余金额', '附言', '现金项目', '转账方式', '最后收款人印章名称', '第一背书印章名称', '业务类型']
}

def downloadShredData(date):
    args = download_args()
    save_dir = args.save_dir
    need_item = clsType2Folder[args.cls_type] # 需要下载的文件夹
    cd_code = "/run/user/1000/gvfs/smb-share:server={},share={}".format(args.server, args.share)

    for item in need_item:
        is_hd = args.is_hd # 是否是手写
        copy_folder_name = date + '_handwriting' if is_hd else date + '_print'

        copy_path = os.path.join(cd_code, item, copy_folder_name)
        copy_list = os.listdir(copy_path)
        if copy_list:
            save_item_dir = os.path.join(save_dir, date, item)
            if not os.path.exists(save_item_dir):
                os.makedirs(save_item_dir)
            print("Date: %s Item: %s Type: %s \t IsHd: %s \t Size: %s \n" %(date, item, args.cls_type, is_hd, len(copy_list)))
            os.system("cd {};cp -r {}/{}/* {}".format(cd_code, item, copy_folder_name, save_item_dir))
        else:
            print("=====>>Warring: No Data!!!\nDate: %s Item: %s Type: %s \t IsHd: %s \t Size: %s \n" %(date, item, args.cls_type, is_hd, len(copy_list)))
        
        
        # os.system("gnome-terminal -e 'bash -c \"cd /run/user/1000/gvfs/smb-share:server=192.168.20.252,share=cls_item;cp -R {} {};exec bash\"'".format(root1,args.output_path))
        

if __name__ == '__main__':
    args = download_args()
    if args.cfg_path:
        cfg_from_file(args.cfg_path)
    date = args.date
    key_dates = [date] if ',' not in date else date.split(',') if type(date) == type('Hi') else date
    for key_date in key_dates:
        downloadShredData(key_date)