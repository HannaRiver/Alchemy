#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import glob
import re


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def isMatch(test_list, img_path):
    for item in test_list:
        date, _, label = item.split('/')[-3: ]
        pattern = '(.*)' + date + '_(.*)_'+ label
        flag = re.match(pattern, img_path)
        if flag:
            os.remove(img_path)
            print(date, label)
            return True
    return False

def get_ca_datalist_labeled():
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/CA'
    save_datalist_root = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/data/V0.1.0'
    ssd_version = 'v001'
    item_list = ['CA', 'Ck', 'DR']
    date_list = ['20180731', '20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']

    f_train_w = open(os.path.join(save_datalist_root, 'train_labeled.txt'), 'w')
    f_test_w = open(os.path.join(save_datalist_root, 'test_labeled.txt'), 'w')

    for item in item_list:
        for date in date_list:
            roi_data_dir = os.path.join(data_root, item, 'labeled', date, ssd_version+'_ssd_roi')

            img_name_list = glob.glob(os.path.join(roi_data_dir, '*.png'))
            lsize = len(img_name_list)
            train_list = img_name_list[: int(0.8*lsize)]
            test_list = img_name_list[int(0.8*lsize): ]

            for img_path in train_list:
                f_train_w.write(img_path + '\n')
            
            for test_path in test_list:
                f_test_w.write(test_path + '\n')
            
    f_train_w.close()
    f_test_w.close()

def get_ca_datalist():
    data_root = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/rspy/CA/'
    save_datalist_root = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/data/V0.1.2'
    ssd_version = 'v001'
    item_list = ['CA', 'Ck', 'DR']
    # date_list = ['20180711', '20180720', '20180724', '20180725', '20180726', '20180727', '20180730',
    #              '20180906', '20180907', '20180910', '20180911', '20180913', '20180914', '20180917', '20180918', '20180919', '20180920', '20180925',
    #              '20181009', '20181011', '20181015', '20181018', '20181022', '20181024'] # , '20181026', '20181029']
    date_list = ['20180731', '20180802', '20180803', '20180808', '20180813', '20180814', '20180816', '20180827', '20180828', '20180829']
    # date_list = ['20180726', '201810022', '20181024']

    f_train_w = open(os.path.join(save_datalist_root, 'train_labeled.txt'), 'w')
    f_test_w = open(os.path.join(save_datalist_root, 'test_labeled.txt'), 'w')

    for item in item_list:
        for date in date_list:
            # roi_data_dir = os.path.join(data_root, item, date, ssd_version+'_ssd_roi')
            roi_data_dir = os.path.join(data_root, item, 'labeled', date, ssd_version+'_ssd_roi')

            img_name_list = glob.glob(os.path.join(roi_data_dir, '*.png'))
            lsize = len(img_name_list)
            # train_list = img_name_list
            train_list = img_name_list[: int(0.8*lsize)]
            test_list = img_name_list[int(0.8*lsize): ]

            for img_path in train_list:
                f_train_w.write(img_path + '\n')
            
            for test_path in test_list:
                f_test_w.write(test_path + '\n')
            
    f_train_w.close()
    f_test_w.close()

def tmp_function():
    data_dir = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/Train/Cdate/cd_rec_v003_test'
    save_dir = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_date/data/V0.0.3/11.txt'
    test_dir = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/data/V0.1.0/org/test.txt'

    fw = open(save_dir, 'w')
    img_path_list = glob.glob(os.path.join(data_dir, '*.png'))
    # test_list = readTxt(test_dir)
    for item in img_path_list:
        # if not isMatch(test_list, item):
        if True:
            fw.write(item + '\n')
    fw.close()


if __name__ == '__main__':
    tmp_function()


