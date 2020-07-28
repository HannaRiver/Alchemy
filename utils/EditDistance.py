#/usr/bin/python3
#-*- encoding=utf-8 -*-

import numpy as np
from collections import Counter
import difflib
import matplotlib.pyplot as plt

def Min_Edit_Distance(source, target, D=[]):
    '''
    source: 模型识别结果
    target: label
    '''
    result = difflib.ndiff(source, target)
    results = [i for i in result]
    for i, item in enumerate(results):
        if item[0] == ' ':
            continue
        if item[0] == '-' and i < len(results) - 1 and results[i+1][0] == '+':
            key = results[i+1][2] + '_' + item[2]
            D.append(key)
            continue
        if item[0] == '+' and i > 0 and results[i-1][0] == '+':
            continue
        if item[0] == '-':
            key = ' _' + item[2]
        if item[0] == '+':
            key = item[2] + '_ '
        D.append(key)
    return D

def MinEditDistance(source, target, D=[]):
    '''
    source: 模型识别结果
    target: label
    '''
    s = difflib.SequenceMatcher(None, source, target)
    for tag, source_start, source_end, target_start, target_end in s.get_opcodes():
        if tag == 'replace':
            source_part = source[source_start: source_end]
            target_part = target[target_start: target_end]
            for i in range(min(len(source_part),len(target_part))):
                D.append(target_part[i] + '_' + source_part[i])
            if len(source_part) > len(target_part):
                for item in source_part[len(target_part): ]:
                    D.append(' _' + item)
            elif len(source_part) < len(target_part):
                for item in target_part[len(source_part): ]:
                    D.append(item + '_ ')
            continue
        elif tag == 'delete':
            source_part = source[source_start: source_end]
            for i in range(len(source_part)):
                D.append(' _' + source_part[i])
            continue
        elif tag == 'insert':
            target_part = target[target_start: target_end]
            for i in range(len(target_part)):
                D.append(target_part[i] + '_ ')
            continue
    return D

def parse_lstm_log(txt_path):
    D = []
    key = "零"
    _IsDeug_ = False
    count = 0
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            if line[0] != '[':
                continue
            target, source = line.strip().split('_')[-1].split('.png ')
            if key not in source and key in target and _IsDeug_: 
                print(line)
            D = MinEditDistance(source, target, D)
            count += 1
    return D, count

def parse_lstm_log_cpp(txt_path):
    D = []
    key = "元"
    count = 0
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            target, source = line.strip().split(',')[-2: ]
            if key not in source and key in target and True: 
                print(line)
            D = MinEditDistance(source, target, D)
            count += 1
    return D, count

def parse_ocrBase_log(ocrbaselog_path, D=[]):
    info_dict = {}
    with open(ocrbaselog_path, 'r') as f:
        for line in f.readlines():
            if '==========' not in line or ': gt: ' not in line:
                continue
            temp_infos = line.strip().split(': gt: ')
            item = temp_infos[0].split('========== ')[-1]
            temp_infos_2 = temp_infos[1].split(' [')
            target = temp_infos_2[0]
            source = temp_infos_2[1].split('-> pre: ')[-1]
            item_D = [] if item not in info_dict else info_dict[item]
            item_D = MinEditDistance(source, target, item_D)
            info_dict[item] = item_D
    return info_dict

def draw_edit_distance(info_dict):
    def draw_single_item(item, D):
        most_common = Counter(D).most_common()
        print("%s[target -> source]:%s\n" %(item, str(most_common)))
        keywords=[item[0] for item in most_common]
        weights=[item[1] for item in most_common]
        axes1 = plt.subplot(111)
        plt.plot(keywords, weights)
        axes1.set_title(item, fontproperties='')
        plt.show()
    AllD = []
    for item in info_dict:
        D = info_dict[item]
        AllD += D
        draw_single_item(item, D)
    draw_single_item("所有类型", AllD)

def main():
    log_path = './tests/testlog/ocr_base_example.log'
    info_dict = parse_ocrBase_log(log_path)
    draw_edit_distance(info_dict) 

if __name__ == '__main__':
    main()
