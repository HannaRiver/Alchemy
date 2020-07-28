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

        chn_tab_fr = open('./fulword_wordslib.pkl', 'rb')
        self.Bank = pickle.load(chn_tab_fr)
        print(len(self.Bank))
        self.Bank_nummax = 30
    
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

def imgname2label(name_list_path, savepath, num_tab, nummax):
    new_num_tab = num_tab
    black_id = num_tab.index(' ')

    img_name_list = readTxt(name_list_path)

    fw = open(savepath, 'w')

    for item in img_name_list:
        # print(item)
        label = os.path.basename(item).split('_')[0]
        cnt = 0

        for i in label:
            # if i == '百':
            #     i = '佰'
            # if i == '萬':
            #     i = '万'
            # if i == '千':
            #     i = '仟'
            if i not in num_tab:
                new_num_tab.append(i)
                print(i)
            cnt += 1
            if cnt == 1:
                fw.write(str(new_num_tab.index(i)))
                continue
            fw.write(' ' + str(new_num_tab.index(i)))
        fw.write((' '+str(black_id)) * (nummax - cnt) + '\n')
    fw1 = open('./fulword_wordslib.pkl', 'wb')
    pickle.dump(new_num_tab, fw1)
    print(len(new_num_tab))

    fw1.close()
    fw.close()





if __name__ == '__main__':
    name_list_path = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_date/data/V0.0.3/11.txt'
    label_savepath = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_date/data/V0.0.3/111.txt'
    combi_savepath = '/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_date/data/V0.0.3/v012_car_train_data.txt'
    mklabel = LabelTab()
    # mklabel.imgname2label_CA(name_list_path, savepath)
    mklabel.imgname2label_CDate(name_list_path, label_savepath)

    os.system("paste -d' ' %s %s >> %s" %(name_list_path, label_savepath, combi_savepath))

    # paste -d' ' 11.txt 111.txt >> v012_car_train_data.txt