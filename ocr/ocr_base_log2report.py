#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import json
import csv
import sys
'''
要求log中有效信息第一行为'\n', 最后一行也为'\n'
目前支持的解析类型为"驾驶证, 行驶证, 转账支票"
'''


logname2jsonname = {
    "转账支票":{
        "出票人账号": 'chupiaorenzhanghao',
        "密码": 'mima',
        "行号": 'hanghao',
        "收款人": 'shoukuanren'},
    "驾驶证":{
        "准驾车型": "chexing",
        "初次领证日期": "chucilingzhengriqi",
        "出生日期": "chushengriqi",
        "国籍": "guoji",
        "性别": "xingbie",
        "姓名": "xingming",
        "有效期限_起始时间": "youxiaoqiixian1",
        "有效期限_终止时间": "youxiaoqiixian2",
        "证号": "zhenghao",
        "住址": "zhuzhi"
    }
}
Item2LastID = {
    "行驶证": 297,
    "合格证": 120,
}

def strQ2B(ustring):
    '''
    全角转换为半角(2/3版本不兼容)
    '''
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def log2dict(log_path, key_char):
    infos = {}
    with open(log_path, 'r') as f:
        info = {}
        flag_pass = True
        for line in f.readlines():
            if '/' not in line and key_char not in line:
                if line == '\n':
                    if flag_pass:
                        flag_pass = False
                        continue
                    else:
                        infos[key_id] = info
                        info = {}
                else:
                    continue
            if line[0] == '/':
                key_id = os.path.splitext(os.path.basename(line.strip()))[0]
            if line[0] == key_char:
                strip_info = line.strip().split('      ')
                a = strip_info[0]
                b = strip_info[1] if len(strip_info) == 2 else ''
                if '证_' in a:
                    info[a.split('证_')[1]] = b
                else:
                    info[a.split('_')[1]] = b
    return infos

def json2value(json_path, json_key):
    if json_key == "zhuzhi":
        gt_item = ''
        with open(json_path, 'r') as json_f:
            data = json.load(json_f)
            obj = data[json_key]
            for adds in obj:
                gt_item += adds[json_key]
        return gt_item
    
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        gt_item = data[json_key][0][json_key]
    return gt_item

def csv2dict(csv_path, img_id_b='297'):
    '''
    img_id_b: csv文件中最后一张图片的id(一般为图片名)
    '''
    gt_infos = {}
    with open(csv_path) as csvfile:
        readCSV = csv.reader(csvfile)
        gt_info = {}
        for row in readCSV:
            img_id = row[0]
            if img_id == img_id_b:
                img_item = row[1]
                img_gt = row[2]
                gt_info[img_item] = img_gt
            else:
                gt_infos[img_id_b] = gt_info
                img_id_b = img_id
                gt_info = {}
                img_item = row[1]
                img_gt = row[2]
                gt_info[img_item] = img_gt
        gt_infos[img_id_b] = gt_info
    return gt_infos

def get_acc4json(regc_item, json_dir, infos, test_item="驾驶证"):
    date_item_list = ["初次领证日期", "有效期限_终止时间", "有效期限_起始时间", "出生日期"]
    for item in regc_item:
        cnt, cnt_r = 0, 0
        for i in infos:
            cnt += 1
            if item not in infos[i]:
                continue
            pre_item = infos[i][item]
            json_key = logname2jsonname[test_item][item]
            json_path = os.path.join(json_dir, i + '.json')
            gt_item = json2value(json_path, json_key)
            if item in date_item_list:
                gt_item = gt_item.replace('-', '')
            if gt_item == pre_item:
                cnt_r += 1
            else:
                # print("网络识别结果为: %s --> gt:: %s" %(pre_item, gt_item))
                pass
        if cnt == 0:
            print(infos)
        else:
            print("Deal with %s cnt:: %s, cnt_r:: %s, acc: %s\n" %(item, cnt, cnt_r, cnt_r/cnt))
    
    cnt_all = 0
    for i in infos:
        cnt_tmp = 0
        for item in infos[i]:
            if item not in regc_item:
                continue
            pre_item = infos[i][item]
            json_path = os.path.join(json_dir, i + '.json')
            json_key = logname2jsonname[test_item][item]
            gt_item = json2value(json_path, json_key)
            if item in date_item_list:
                gt_item = gt_item.replace('-', '')
            if gt_item == pre_item:
                cnt_tmp += 1
        if cnt_tmp == len(regc_item):
            cnt_all += 1
    print("表数:: %s, 整表识别正确:: %s, 整表识别率: %s\n" %(cnt, cnt_all, cnt_all/cnt))

def get_acc4csv(regc_item, csv_path, infos, test_item="行驶证"):
    img_id_b = Item2LastID[test_item]
    gt_infos = csv2dict(csv_path, img_id_b)
    for item in regc_item:
        cnt, cnt_r = 0, 0
        for i in infos:
            cnt += 1
            if item not in infos[i]:
                continue
            model_result = infos[i][item]
            if item in ['注册日期', '发证日期']:
                model_result = model_result.replace('-', '')
            if strQ2B(model_result) == strQ2B(gt_infos[i][item]):
                cnt_r += 1
            else:
                pass
        print("Deal with %s cnt:: %s, cnt_r:: %s, acc: %s\n" %(item, cnt, cnt_r, cnt_r/cnt))
    
    cnt_all = 0
    for i in infos:
        cnt_tmp = 0
        for item in infos[i]:
            if item not in regc_item:
                continue
            pre_label = infos[i][item]
            gt_label = gt_infos[i][item]
            if item in ['注册日期', '发证日期']:
                pre_label = pre_label.replace('-', '')
            if pre_label == gt_label:
                cnt_tmp += 1
        # 行驶证有10项
        if cnt_tmp == len(regc_item):
            cnt_all += 1
    print("表数:: %s, 整表识别正确:: %s, 整表识别率: %s\n" %(cnt, cnt_all, cnt_all/cnt))


def test_drivingLicense(log_path, gt_dir):
    '''
    驾驶证解析日志出测试报告代码
    '''
    model_results = log2dict(log_path, '驾')
    regc_item = ["证号", "国籍", "姓名", "性别", "住址", "出生日期", "初次领证日期", "准驾车型", "有效期限_终止时间", "有效期限_起始时间"]
    get_acc4json(regc_item, gt_dir, model_results, "驾驶证")

def test_vehicleLicense(log_path, gt_dir):
    '''
    行驶证解析日志出测试报告
    '''
    model_results = log2dict(log_path, '行')
    regc_item = ['号牌号码', '车辆类型', '所有人', '使用性质', '品牌型号', '车辆识别代码', '发动机号码', '注册日期', '发证日期', '住址']
    get_acc4csv(regc_item, gt_dir, model_results, "行驶证")

def test_transferCheque(log_path, gt_dir):
    '''
    转账支票日志出测试报告
    '''
    model_results = log2dict(log_path, '支')
    # regc_item = ['出票日期', '付款行名称', '收款人', '出票人账号', '人民币大写', '人民币小写', '用途', '密码', '行号']
    regc_item = ['出票人账号', '密码', '行号', '收款人']
    get_acc4json(regc_item, gt_dir, model_results, "转账支票")

def test_carCertificate(log_path, gt_dir):
    '''
    合格证日志出测试报告
    '''
    model_results = log2dict(log_path, '车')
    regc_item = ['车辆识别代号', '车身颜色', '轮胎规格', '车辆制造日期']
    get_acc4csv(regc_item, gt_dir, model_results, "合格证")

def test_IDCard(log_path, gt_dir):
    '''
    身份证日志出测试报告
    '''
    model_results = log2dict(log_path, '身')
    regc_item = ['姓名', '性别', '民族', '出生', '住址', '身份证号码']
    get_acc4csv(regc_item, gt_dir, model_results, "身份证")

def log2report_help():
    print('''
             脚本作用: 解析工程日志 --> 各项准确度报告\n
             Python Versin: 3.0+\n
             支持类型: 驾驶证、行驶证、转账支票、合格证\n
             使用: python3 ocr_base_log2report.py type[如"驾驶证"/"0"] log_path \n
             注意事项:要求log中有效信息第一行为换行, 最后一行也为换行, 如果是工程直接输出结果，在log最后一行加一个换行就行''')

def ocr_base_log2report(log_path, test_item):
    print("Deal with the %s info... " %(test_item))
    if test_item == '驾驶证':
        gt_dir = '/work/ocr/pro/CommonOCR/testImg/驾驶证/标准答案'
        test_drivingLicense(log_path, gt_dir)
    if test_item == '转账支票':
        gt_dir = '/work/ocr/pro/CommonOCR/testImg/转账支票/标准答案'
        test_transferCheque(log_path, gt_dir)
    if test_item == '行驶证':
        gt_dir = '/work/ocr/card/vehicle_license/data/test/info.csv'
        test_vehicleLicense(log_path, gt_dir)
    if test_item == '合格证':
        gt_dir = '/work/ocr/pro/CommonOCR/testImg/合格证/info.csv'
        test_carCertificate(log_path, gt_dir)
    if test_item == '身份证':
        gt_dir = '/work/ocr/pro/CommonOCR/testImg/身份证/info.csv'
        test_IDCard(log_path, gt_dir)

if __name__ == '__main__':
    test_items = ['驾驶证', '行驶证', "转账支票", "合格证", "身份证"]
    if len(sys.argv) == 1:
        log2report_help()
        test_item = test_items[0]
        log_path = '/work/ocr/pro/CommonOCR/bin/drivingLicense_20190329'
        ocr_base_log2report(log_path, test_item)
    else:
        test_id = sys.argv[1]
        if len(test_id) == 1:
            test_item = test_items[int(test_id)]
        elif test_id in test_items:
            test_item = test_id
        else:
            log2report_help()
            sys.exit()
        log_path = sys.argv[2]
        ocr_base_log2report(log_path, test_item)
    