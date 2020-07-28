#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..', '..', '..'))
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

        # chn_tab_fr = open('', 'rb')
        #self.Bank = pickle.load(chn_tab_fr)
        #print(len(self.Bank))
        #self.Bank_nummax = 30
    
    def imgname2label_CA(self, name_list_path, savepath, indx):
        '''
        大写金额 根据文件名转换为训练所需的标签
        '''
        imgname2label(name_list_path, savepath, self.CA, self.CA_nummax, indx)
        
    def imgname2label_CDate(self, name_list_path, savepath, indx):
        '''
        大写日期， 根据文件名转换为训练所需要的标签
        '''
        imgname2label(name_list_path, savepath, self.CDate, self.CDate_nummax, indx)

def imgname2label(name_list_path, savepath, num_tab, nummax, indx=0):
    new_num_tab = num_tab
    black_id = num_tab.index(' ')

    img_name_list = readTxt(name_list_path)

    fw = open(savepath, 'w')

    for item in img_name_list:
        # print(item)
        label = os.path.splitext(os.path.basename(item))[0].split('_')[indx]
        cnt = 0
        for i in label:
            if i not in num_tab:
                new_num_tab.append(i)
                print(i)
            cnt += 1
            if cnt == 1:
                fw.write(str(new_num_tab.index(i)))
                continue
            fw.write(' ' + str(new_num_tab.index(i)))
        fw.write((' '+str(black_id)) * (nummax - cnt) + '\n')
    fw1 = open(os.path.join(os.path.dirname(sys.argv[2]), 'ca_wordslib.pkl'), 'wb')
    pickle.dump(new_num_tab, fw1)
    print(len(new_num_tab))

    fw1.close()
    fw.close()

if __name__ == '__main__':
    name_list_path = sys.argv[1]
    label_savepath = './need2del.txt'
    combi_savepath = sys.argv[2]
    indx = int(sys.argv[3])
    mklabel = LabelTab()
    mklabel.imgname2label_CA(name_list_path, label_savepath, indx)
    # mklabel.imgname2label_CDate(name_list_path, label_savepath)

    os.system("paste -d' ' %s %s >> %s" %(name_list_path, label_savepath, combi_savepath))

    # paste -d' ' 11.txt 111.txt >> v012_car_train_data.txt





