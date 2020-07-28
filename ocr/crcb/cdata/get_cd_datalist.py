#/usr/bin/python3
#-*- encoding=utf-8 -*-

import os
import glob


def get_cd_datalist():
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/Date/CDate'
    save_datalist_root = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_date/data/V0.0.1'
    deter_version = ''
    item_list = ['Ck', 'Voucher']

    tt_ratio = 0.8
    cd_train_file = open(os.path.join(save_datalist_root, 'train_org.txt'), 'w')
    cd_test_file = open(os.path.join(save_datalist_root, 'test_org.txt'), 'w')
    cd_train_file_add = open(os.path.join(save_datalist_root, 'train_add.txt'), 'w')
    cd_test_file_add = open(os.path.join(save_datalist_root, 'test_add.txt'), 'w')

    for item in item_list:
        date_list = os.listdir(os.path.join(data_root, item))
        for date in date_list:
            roi_data_dir = os.path.join(data_root, item, date, 'roi')
            ssd_roi_dir = os.path.join(data_root, item, date, deter_version + 'ssd_roi')

            if not os.path.exists(roi_data_dir):
                print("Path: %s not exists!" %(roi_data_dir))
                continue

            img_name_list = glob.glob(os.path.join(roi_data_dir, '*.png'))

            lsize = len(img_name_list)
            train_list = img_name_list[: int(tt_ratio*lsize)]
            test_list = img_name_list[int(tt_ratio*lsize): ]

            flag = os.path.exists(ssd_roi_dir)

            if not flag:
                print("Path: %s not exists!" %(ssd_roi_dir))

            for img_path in train_list:
                cd_train_file.write(img_path + '\n')
                ssd_roi_path = os.path.join(ssd_roi_dir, os.path.basename(img_path))
                if flag:
                    if os.path.exists(ssd_roi_path):
                        cd_train_file_add.write(ssd_roi_path + '\n')
                    if os.path.exists(ssd_roi_path + '.png'):
                        cd_train_file_add.write(ssd_roi_path + '.png' + '\n')

            for img_path in test_list:
                cd_test_file.write(img_path + '\n')
                ssd_roi_path = os.path.join(ssd_roi_dir, os.path.basename(img_path))
                if flag and os.path.exists(ssd_roi_path):
                    cd_test_file_add.write(ssd_roi_path + '\n')

    cd_train_file.close()
    cd_test_file.close()            
    cd_train_file_add.close()
    cd_test_file_add.close()

if __name__ == '__main__':
    get_cd_datalist()

