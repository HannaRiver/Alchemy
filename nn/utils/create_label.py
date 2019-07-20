#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
from file2list import readTxt
import pickle


class LabelTab:
    def __init__(self):
        self.CA = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
                   '拾', '元', '佰', '仟', '万', '亿', '角', '分', '整', '正',
                   '圆', ' ']
        self.CA_nummax = 20

        self.CDate = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
                      '拾', '年', '月', '日', ' ']
        self.CDate_nummax = 12

        chn_tab_fr = open('/work/ocr/card/driving_license/model/V0.0.0/data/fulword_wordslib.pkl', 'rb')
        self.Bank = pickle.load(chn_tab_fr)
        # print(len(self.Bank))
        self.Bank_nummax = 30

        self.SROIE = [' ','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.SROIE_nummax = 68

        self.Money = [' ']

        chn_tab_fr = open('/work/ocr/certificate/car_certificate/model/pro/fixed_item_recog_tab_hgz.pkl', 'rb')
        self.Coc = pickle.load(chn_tab_fr)
        self.Coc_nummax = 30

        self.Model = [' '] + readTxt('/work/ocr/card/vehicle_license/model/rec/tonana/all0624_wordlib.txt')
        self.Model_nummax = 37
    
    def imgname2label_CA(self, name_list_path, savepath):
        '''
        大写金额 根据文件名转换为训练所需的标签
        '''
        imgname2label(name_list_path, savepath, self.CA, self.CA_nummax)
        
    def imgname2label_CDate(self, name_list_path, savepath):
        '''
        大写日期， 根据文件名转换为训练所需要的标签
        '''
        imgname2label(name_list_path, savepath, self.CDate, self.CDate_nummax)
    
    def imgname2label_Bank(self, name_list_path, savepath):
        '''
        银行非固定项， 根据文件名转换为训练所需要的标签
        '''
        imgname2label(name_list_path, savepath, self.Bank, self.Bank_nummax)
    
    def imgname2label_SROIE(self, name_list_path, savepath):
        '''
        SROIE task2
        '''
        imgname2label(name_list_path, savepath, self.SROIE, self.SROIE_nummax)
    
    def imgname2label_Coc(self, name_list_path, savepath):
        '''
        一致性证书， 根据文件名转换训练所需的标签
        '''
        imgname2label(name_list_path, savepath, self.Coc, self.Coc_nummax)
    
    def imgname2label_Model(self, name_list_path, savepath):
        '''
        Model
        '''
        imgname2label(name_list_path, savepath, self.Model, self.Model_nummax)


def imgname2label(name_list_path, savepath, num_tab, nummax):
    new_num_tab = num_tab
    black_id = num_tab.index(' ')

    img_name_list = readTxt(name_list_path)

    fw = open(savepath, 'w')

    for item in img_name_list:
        label = os.path.splitext(os.path.basename(item))[0].split('_')[0]
        # if len(label) != 17:
        #     print(item)
        #     break
        cnt = 0
        for i in label:
            # if i == '百':
            #     i = '佰'
            # if i == '萬':
            #     i = '万'
            # if i == '千':
            #     i = '仟'
            # i = ' ' if i == '→' else i
            # i = '/' if i == '↑' else i
            if i == '\u200c':
                print(item)
            i = '(' if i == '（' else i
            i = ')' if i == '）' else i
            i = '!' if i == '！' else i
            i = '#' if i == '❶' else i
            i = '.' if i == '﹒' else i
            # i = i.upper()
            if i not in num_tab:
                new_num_tab.append(i)
            cnt += 1
            if cnt == 1:
                fw.write(str(new_num_tab.index(i)))
                continue
            fw.write(' ' + str(new_num_tab.index(i)))
        fw.write((' '+str(black_id)) * (nummax - cnt) + '\n')
    fw1 = open('/work/ocr/card/vehicle_license/model/rec/tonana/model.pkl', 'wb')
    pickle.dump(new_num_tab, fw1)
    print(new_num_tab)
    print(len(new_num_tab))

    fw1.close()
    fw.close()

if __name__ == '__main__':
    name_list_path = '/work/ocr/card/vehicle_license/model/rec/tonana/data/test_no.txt'
    label_savepath = '/work/ocr/certificate/car_certificate/model/add_coc_color/data/tmp.txt'
    combi_savepath = '/work/ocr/card/vehicle_license/model/rec/tonana/data/test_no_label.txt'
    mklabel = LabelTab()
    # mklabel.imgname2label_CA(name_list_path, savepath)
    mklabel.imgname2label_Model(name_list_path, label_savepath)

    os.system("paste -d' ' %s %s >> %s" %(name_list_path, label_savepath, combi_savepath))
    # os.system("paste -d' ' %s %s >> %s" %('/work/competitions/ICDAR/SROIE/task2/data/train.txt', label_savepath, combi_savepath))
    # paste -d' ' 11.txt 111.txt >> v012_car_train_data.txt