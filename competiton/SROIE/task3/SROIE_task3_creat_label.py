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

def task12wordinfo(txtpath):
    '''
    将任务1中的标签转换为单词列表结构为:
    [['43,20,351,20,351,71,43,71,TAN WOON YANN','TAN', 'WOON', 'YANN'],[','], ..., []]
    '''
    filelist = readTxt(txtpath)
    # 对结果进行位置从上到下从左到右的顺序进行排序操作
    filelist.sort(key=lambda x: (sum([x[i] for i in [1, 3, 5, 7]]), sum([x[i] for i in [0, 2, 4, 6]])))

    save_path = os.path.join('/work/competitions/ICDAR/SROIE/data/task2_pre', os.path.basename(txtpath))
    save_img_path = os.path.join('/work/competitions/ICDAR/SROIE/data/task2_img_pre', os.path.basename(txtpath)[: -3] + 'jpg')
    img_path = txtpath[: -3] + 'jpg'
    img = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fw = open(save_path, 'w')
    for i,line in enumerate(filelist):
        x = line
        color = (0, 0, 255)
        ymin = min(x[1], x[3], x[5], x[7])
        ymax = max(x[1], x[3], x[5], x[7])
        xmin = min(x[0], x[2], x[4], x[6])
        xmax = max(x[0], x[2], x[4], x[6])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.putText(img, (str(i)), (xmin - 5, ymin - 5), font, 0.6, (0, 255, 0), 1)
        fw.write(','.join([str(i) for i in line]) + '\n')
    fw.close()
    cv2.imwrite(save_img_path, img)
    return 

    
    result = []
    for line in filelist:
        word_segments = get_task1_word_segment(line[-1])
        result.append([line] + word_segments)
    return result
    


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            locates = [int(i) for i in line.strip().split(',')[: 8]]
            locate = ','.join(line.strip().split(',')[: 8])
            content = line.strip()[len(locate)+1: ]
            locates.append(content)
            filelist.append(locates)
    return filelist

def WordTokenization(sentence, lang='eng'):
    result = nltk.word_tokenize(sentence) if lang == 'eng' else jieba.cut(sentence)
    return result if lang == 'eng' else [i for i in result]

def get_locate_cls_info(txtpath, json_path, save_path):
    '''
    43,20,351,20,351,71,43,71,TAN WOON YANN ->
    O, 43,20,351,20,351,71,43,71,TAN WOON YANN
    O: Others
    C: Company
    D: Date
    A: Address
    T: Total
    '''
    # 获取按位置排序后的文本信息行包括分词结果
    task1info = task12wordinfo(txtpath)
    # 获取信息字典
    # task2info = task22wordinfo(json_path)
    # named_entity_info = NamedEntityLabeled()
    pass

def get_single_info(txtpath, json_path, save_type, save_path):
    '''
    txtpath: task1标注的位置及识别结果的txt文档
    json_path: task2标注的提取的信息字典的txt文档
    save_type: 导出的文本类型，目前支持[locate_cls, ner_sentence, ner_word]
    '''
    if save_type == 'locate_cls':
        get_locate_cls_info(txtpath, json_path, save_path)
    elif save_type == 'ner_sentence':
        get_ner_sentence_info(txtpath, json_path, save_path)
    elif save_type == 'ner_word':
        get_ner_word_info(txtpath, json_path, save_path)
    else:
        ValueError('save type error! Support locate_cls, ner_sentence, ner_word.')

def main(txt_dir, json_dir, save_dir, save_type):
    if os.path.isfile(txt_dir):
        file_name = os.path.basename(save_dir)
        save_path = os.path.join(save_dir, file_name)
        get_single_info(txt_dir, json_dir, save_type, save_path)
        return
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for item in os.listdir(txt_dir):
        if item[-3: ] != 'txt':
            continue
        else:
            shutil.copy(os.path.join(txt_dir, item[: -4] + '.jpg'), os.path.join('/work/competitions/ICDAR/SROIE/data/task1-task2', item[: -4] + '.jpg'))
            continue
        txt_path = os.path.join(txt_dir, item)
        json_path = os.path.join(json_dir, item)
        if not os.path.exists(json_path):
            continue
        save_path = os.path.join(save_dir, item)
        get_single_info(txt_path, json_path, save_type, save_path)


def get_task1_word_segment(content):
    # locate = ','.join(line.split(',')[: 8])
    # content = line[len(locate)+1: ]
    if not content:
        return content
    lang = 'chn' if ord(content[0]) > 255 else 'eng'
    word_segments = WordTokenization(content, lang)
    return word_segments

def NamedEntityLabeled(content, dict_info):
    word_segments = get_task1_word_segment(content)
    company_segments = get_task1_word_segment(dict_info["company"]) if "company" in dict_info else set()
    if set(word_segments) & set(company_segments):
        return 'C,'
    date = dict_info["date"] if "date" in dict_info else "我也很无奈写这个"
    if date in content:
        return 'D,'
    address_segements = get_task1_word_segment(dict_info["address"]) if "address" in dict_info else set()
    if set(word_segments) & set(address_segements):
        return 'A,'
    total = dict_info["total"] if "total" in dict_info else "你猜猜"
    if total in content:
        return "T,"
    return "O,"
    

def tmp(txtpath, save_path, dict_info):
    fw = open(save_path, 'w')
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            locates = [int(i) for i in line.strip().split(',')[: 8]]
            locate = ','.join(line.strip().split(',')[: 8])
            content = line.strip()[len(locate)+1: ]
            named_entity = NamedEntityLabeled(content, dict_info)
            fw.write(named_entity+line)
    fw.close()

def tmp_main(txt_dir, json_dir, save_dir):
    for item in os.listdir(txt_dir):
        txt_path = os.path.join(txt_dir, item)
        json_path = os.path.join(json_dir, item)
        save_path = os.path.join(save_dir, item)
        dict_info = task22wordinfo(json_path)
        tmp(txt_path, save_path, dict_info)

if __name__ == '__main__':
    save_dir = '/work/competitions/ICDAR/SROIE/data/task2_pre_2/'
    json_dir = '/work/competitions/ICDAR/SROIE/data/task2_train/'
    txt_dir = '/work/competitions/ICDAR/SROIE/data/task2_pre'
    # main(txt_path, json_path, save_dir, 'locate_cls')
    tmp_main(txt_dir, json_dir, save_dir)

