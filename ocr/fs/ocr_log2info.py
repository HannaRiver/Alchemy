#/usr/bin/python3
#-*- encoding=utf-8 -*-
import os
import sys
import argparse
import logging
from functools import reduce
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))

from char_trans import strQ2B
from char_trans import MyIsdigit
from char_trans import rmPunctuation
from char_trans import rmParenthesesItem
from file2list import readTxt

sh_path = os.path.split(os.path.realpath(__file__))[0]
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def NormalTestWay(test_way):
    if type(test_way) == str or type(test_way) == int:
        if len(test_way) == 1:
            return [int(test_way)]
        return test_way.split(',')
    else:
        assert(type(test_way) == list), 'test_way must be a list or string by ,'
        return test_way

def NormalItem(astr):
    return astr

def test_way2funtion(test_way):
    function_list = [NormalItem, strQ2B, rmPunctuation, rmParenthesesItem]
    return [function_list[int(i)] for i in test_way]

def OCRlog2Info(args):
    fw = open(os.path.join(sh_path, 'save_error_result.txt'), 'w')
    whole_cnt, file_cnt = 0, 0
    all_right_cnt, chn_cnt, num_cnt, tmp_right_cnt, tmp_wrong_cnt, table_type = 0, 0, 0, 0, 0, 0
    img_suffix = ['tif', 'png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'bmp']
    log_info = readTxt(args.log_path)
    for i, line in enumerate(log_info):
        # 一个图片的终结
        if 'whole right' in line:
            whole_cnt += 1
            continue
        if line.split(' ')[0] == 'finanstate_api_process':
            if tmp_right_cnt != 0 and tmp_right_cnt == tmp_wrong_cnt:
                all_right_cnt += 1
            tmp_right_cnt, tmp_wrong_cnt = 0, 0
            continue
        # 一张表的开始
        if line.split('.')[-1] in img_suffix:
            logging.info(line)
            file_cnt += 1
            continue
        if line[: 10] == 'table_type':
            table_type = line[-1]
            continue
        if line[: 5] != 'ri:hi':
            continue
        rw_info = line.split('r:::')[-1].split('  wrong::: ')
        if len(rw_info) == 1:
            right, wrong = rw_info[0], ''
        else:
            right, wrong = rw_info
        fw.write(right+' '+wrong+'\n') 
        tmp_wrong_cnt += 1
        re_right, re_wrong = right, wrong
        for func in test_way2funtion(NormalTestWay(args.test_way)):
            re_right = func(re_right)
            re_wrong = func(re_wrong)
        logging.info('right: %s(%s), wrong: %s(%s)' %(right, re_right, wrong, re_wrong))
        if re_right == re_wrong:
            tmp_right_cnt += 1
            if MyIsdigit(re_right):
                logging.info("========== IS Digit :: True ==========")
                num_cnt += 1
            else:
                logging.info("========== IS Chn :: True ==========")
                chn_cnt += 1

    fw.close()
    # right_num = int(log_info[-4][11: ])
    chn_right, chn_wr = [int(log_info[-3].split(' ')[i]) for i in [2, 4]]
    num_right, num_wr = [int(log_info[-2].split(' ')[i]) for i in [2, 5]]
    logging.info("=========================================================")
    logging.info("原始状态 :: 中文单条准确率: %s, 数字整条准确率: %s, 整表准确度: %s" %(chn_right/float(chn_right+chn_wr), num_right/float(num_right+num_wr), whole_cnt/float(file_cnt)))
    logging.info("模糊匹配组合方式为: " +  str(args.test_way))
    logging.info("新增整表识别正确：%s, 中文： %s, 数字: %s" %(all_right_cnt, chn_cnt, num_cnt))
    logging.info("现在状态 :: 中文单条准确率: %s, 数字整条准确率: %s, 整表准确度: %s" %((chn_right+chn_cnt)/float(chn_right+chn_wr), (num_right+num_cnt)/float(num_right+num_wr), (whole_cnt+all_right_cnt)/float(file_cnt)))
        



def main(args):
    OCRlog2Info(args)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', help='ocr pro logs path',
                        default=os.path.join(sh_path, 'ocr_log_case.txt'))
    parser.add_argument('--test_way', default=[0], 
                        help='Tips:You can enter a combination 0,1,2,3 \n 0:normal; 1:Convert full angle; 2:Do not compare . with ,; 3:Dont (~);')
    parser.add_argument('--is_mapping', type=int, default=0, help='1: Bayes Way; 2: QA Way')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())