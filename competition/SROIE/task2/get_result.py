#/usr/bin/python3
#-*- coding=utf-8 -*-
import os
import json
import nltk
import jieba
import cv2
import shutil


def task22wordinfo(json_path):
    with open(json_path, 'r') as json_f:
        results = {}
        data = json.load(json_f)
        for item in ["company", "date", "address", "total"]:
            if item not in data:
                continue
            obj = data[item]
            results[item] = obj
    return results

def WordTokenization(sentence, lang='eng'):
    result = nltk.word_tokenize(sentence) if lang == 'eng' else jieba.cut(sentence)
    return result if lang == 'eng' else [i for i in result]

def get_task1_word_segment(content):
    # locate = ','.join(line.split(',')[: 8])
    # content = line[len(locate)+1: ]
    if not content:
        return content
    lang = 'chn' if ord(content[0]) > 255 else 'eng'
    word_segments = WordTokenization(content, lang)
    return word_segments

def GetSegmentResult(result_txt_path, save_txt_path):
    fw = open(save_txt_path, 'w')
    with open(result_txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            result_list = line.split(',')[9: ]
            result = ','.join([i.upper() for i in result_list])
            # word_segments = get_task1_word_segment(result)
            word_segments = result.split(' ') if ' ' in result else [result]
            for item in word_segments:
                fw.write(item + '\n')
    fw.close()

def NamedEntityLabeled(segment_words, entity, dict_info):
    '''
    segment_word: ['TAN', 'WOON' 'YANN']
    entity: 'O'
    dict_info: {}
    '''
    entitys = ['O'] * len(segment_words)
    if entity == 'O':
        return entitys
    if entity == 'D':
        date_gt = dict_info["date"]
        date_idx = segment_words.index(date_gt) if date_gt in segment_words else 0
        entitys[date_idx] = 'D'
        return entitys
    if entity == 'T':
        if len(segment_words) == 1:
            return ['T']
        return ['T_B'] + ['T_I'] * (len(segment_words)-2) + ['T_E']
        date_gt = dict_info["total"]
        date_idx = segment_words.index(date_gt) if date_gt in segment_words else 0
        entitys[date_idx] = 'T'
        return entitys
    if entity == 'C':
        if len(segment_words) == 1:
            return ['C_E']
        entitys = ['C_B']+['C_I'] * (len(segment_words)-2) + ['C_E']
        companys = dict_info["company"].split(' ') if ' ' in dict_info["company"] else [dict_info["company"]]
        company_b = companys[0]
        company_e = companys[-1]
        flage_begin = False
        flage_end = False
        for i, item in enumerate(segment_words):
            if item == company_b or item in company_b and not flage_begin:
                entitys[i] = 'C_B'
                entitys[: i] = ['O'] * i
                flage_begin = True
                begin_idx = i
                continue
            if item == company_e or item in company_e or company_e in item:
                if i != len(segment_words) -1 and company_e not in segment_words[i+1: ] and company_e not in ''.join(segment_words[i+1: ])+'.':
                    flage_end = True
                    entitys[i] = 'C_E'
                    if not flage_begin:
                        entitys[0] = 'C_B'
                        begin_idx = 0
                    entitys[begin_idx+1: i] = ['C_I']*(i-1-begin_idx)
                    entitys[i+1: ] = ['O']*(len(segment_words) - i - 1)
                    return entitys
                if i == len(segment_words) -1:
                    entitys[i] = 'C_E'
                    if not flage_begin:
                        entitys[0] = 'C_B'
                        begin_idx = 0
                        entitys[begin_idx+1: i] = ['C_I']*(i-1-begin_idx)
                    return entitys
        return entitys
    if entity == 'A':
        if len(segment_words) == 1:
            return ['A_E']
        entitys = ['A_B']+['A_I'] * (len(segment_words)-2) + ['A_E']
        companys = dict_info["address"].split(' ') if ' ' in dict_info["address"] else [dict_info["address"]]
        company_b = companys[0]
        company_e = companys[-1]
        flage_begin = False
        flage_end = False
        for i, item in enumerate(segment_words):
            if item == company_b or item in company_b and not flage_begin:
                entitys[i] = 'A_B'
                entitys[: i] = ['O'] * i
                flage_begin = True
                begin_idx = i
                continue
            if item == company_e or item in company_e or company_e in item:
                if i != len(segment_words) -1 and company_e not in segment_words[i+1: ] and company_e not in ''.join(segment_words[i+1: ])+'.':
                    flage_end = True
                    entitys[i] = 'A_E'
                    if not flage_begin:
                        entitys[0] = 'A_B'
                        begin_idx = 0
                    entitys[begin_idx+1: i] = ['A_I']*(i-1-begin_idx)
                    entitys[i+1: ] = ['O']*(len(segment_words) - i - 1)
                    return entitys
                if i == len(segment_words) -1:
                    entitys[i] = 'A_E'
                    if not flage_begin:
                        entitys[0] = 'A_B'
                        begin_idx = 0
                        entitys[begin_idx+1: i] = ['A_I']*(i-1-begin_idx)
                    return entitys
        return entitys 
    return entitys

def GetEntityLabel(dict_info, line, idx_c, idx_e=0):
    result_list = line.split(',')[idx_c: ]
    content = ','.join([i.upper() for i in result_list]) # TAN WOON YANN
    contents = content.split(' ') if ' ' in content else [content] # ['TAN', 'WOON' 'YANN']
    segment_words = [item for item in [MySegmentWord(contents, i) for i in range(len(contents))] if item != '']
    entity = line.split(',')[idx_e] # 'O'
    entitys = NamedEntityLabeled(segment_words, entity, dict_info)
    return segment_words, entitys

def GetSegmentResult4NER(result_txt_path, save_txt_path, dict_info):
    segment_words_list = []
    entitys_list = []
    with open(result_txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            segment_words, entitys = GetEntityLabel(dict_info, line, 9, 0)
            assert(len(segment_words) == len(entitys)), result_txt_path
            segment_words_list += segment_words
            entitys_list += entitys
    flag_addr = True if 'A_B' in entitys_list or 'A_E' in entitys_list else False
    flag_comp = True if 'C_B' in entitys_list or 'C_E' in entitys_list else False
    
    addr_b_idx = entitys_list.index('A_B') if 'A_B' in entitys_list else entitys_list.index('A_E') if flag_addr else -1
    comp_b_idx = entitys_list.index('C_B') if 'C_B' in entitys_list else entitys_list.index('C_E') if flag_comp else -1
    for i, item in enumerate(entitys_list):
        if item == 'A_E':
            addr_e_idx = i
            if i <= addr_b_idx:
                entitys_list[i] = 'A_B'
                addr_b_idx = i
        if item == 'C_E':
            comp_e_idx = i
            if i <= comp_b_idx:
                entitys_list[i] = 'C_B'
                comp_b_idx = i
    if comp_b_idx != -1:
        entitys_list[comp_b_idx+1: comp_e_idx] = ['C_I'] * (comp_e_idx - (comp_b_idx+1))
    if addr_b_idx != -1:
        entitys_list[addr_b_idx+1: addr_e_idx] = ['A_I'] * (addr_e_idx - (addr_b_idx+1))
    fw = open(save_txt_path, 'w')
    for i in range(len(entitys_list)):
        fw.write(entitys_list[i] + ',' + segment_words_list[i] + '\n')
    fw.close()


def MySegmentWord(contents, i):
    item = contents[i]
    if i != len(contents) - 1 and contents[i+1] == ':':
        fw_content = item + ':'
    elif item == ':':
        return ''
    elif i != len(contents) - 1 and contents[i+1] == '.':
        fw_content = item + '.'
    elif item == '.' and i != 0 and contents[i-1] == 'NO':
        return ''
    elif item == 'GET':
        fw_content = 'GST'
    elif item == 'EOLD':
        fw_content = 'SOLD'
    else:
        fw_content = item
    return fw_content

def TmpFunction(result_txt_path, save_txt_path):
    fw = open(save_txt_path, 'w')
    contents = []
    with open(result_txt_path, 'r') as f:
        for line in f.readlines():
            result_list = line.strip().split(',')[9: ]
            result = ','.join([i.upper() for i in result_list])
            word_segments = result.split(' ') if ' ' in result else [result]
            contents += word_segments
    for i, item in enumerate(contents):
        fw_content = MySegmentWord(contents, i)
        if fw_content == '':
            continue
        fw.write(fw_content + '\n')
    fw.close()

def BatchTmpFunction(result_txt_dir, save_txt_dir):
    if not os.path.exists(save_txt_dir):
        os.mkdir(save_txt_dir)
    for txt_name in os.listdir(result_txt_dir):
        result_txt_path = os.path.join(result_txt_dir, txt_name)
        save_txt_path = os.path.join(save_txt_dir, txt_name)
        TmpFunction(result_txt_path, save_txt_path)  

def BatchGetSegmentResult(result_txt_dir, save_txt_dir):
    if not os.path.exists(save_txt_dir):
        os.mkdir(save_txt_dir)
    for txt_name in os.listdir(result_txt_dir):
        result_txt_path = os.path.join(result_txt_dir, txt_name)
        save_txt_path = os.path.join(save_txt_dir, txt_name)
        GetSegmentResult(result_txt_path, save_txt_path)

def BatchGetSegmentResult4NER(result_txt_dir, json_dir, save_txt_dir):
    if not os.path.exists(save_txt_dir):
        os.mkdir(save_txt_dir)
    for txt_name in os.listdir(result_txt_dir):
        print(txt_name)
        result_txt_path = os.path.join(result_txt_dir, txt_name)
        save_txt_path = os.path.join(save_txt_dir, txt_name)
        json_path = os.path.join(json_dir, txt_name)
        dict_info = task22wordinfo(json_path)
        GetSegmentResult4NER(result_txt_path, save_txt_path, dict_info)

def GetResult4SROIE(result_txt_path, save_txt_path):
    fw = open(save_txt_path, 'w')
    with open(result_txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            result_list = line.split(',')[8: ]
            reuslt = ','.join([i.upper() for i in result_list])
            fw.write(reuslt + '\n')
    fw.close()

def BatchGetResult4SROIE(result_txt_dir, save_txt_dir):
    if not os.path.exists(save_txt_dir):
        os.mkdir(save_txt_dir)
    for txt_name in os.listdir(result_txt_dir):
        result_txt_path = os.path.join(result_txt_dir, txt_name)
        save_txt_path = os.path.join(save_txt_dir, txt_name)
        GetResult4SROIE(result_txt_path, save_txt_path)

if __name__ == '__main__':
    result_txt_dir = '/work/competitions/ICDAR/SROIE/task3/sort_txt'
    save_txt_dir = '/work/competitions/ICDAR/SROIE/task3/sort_segment_0428'
    BatchTmpFunction(result_txt_dir, save_txt_dir)
    

