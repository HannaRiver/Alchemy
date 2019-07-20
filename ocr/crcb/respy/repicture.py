#!/usr/bin/env python
"""
repicture.py is a map contact bin imgname, item and lable.

Author: 8121

Time: 2018.7.25
"""

import os
import sys
import glob
import argparse
import logging
from alchemy_config import cfg

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def asql2img_chip_data(sql_insert):
    """
    Only applies to crcb
     a sql -> imagename, taskxml, resultxml
    """
    tmp_result = [item.split('\\')[-1].split('||')[0] for item in sql_insert.split('${data')[1 :]]
    if len(tmp_result) != 3:
        return '0', str(tmp_result), '0'
    imgname, taskxml, resultxml = tmp_result
    return imgname, taskxml, resultxml

def taskxml2key(taskxml):
    """
    get item info and key to find label from resultxml
    """
    xml_file = open(taskxml, 'r').read()
    key, tmp = xml_file.split('field code="')[-1].split('" name="')
    task_value = tmp.split('" ')[0]
    return key, task_value

def key2result(resultxml, key):
    xml_file = open(resultxml, 'r').read()
    result = xml_file.split(key)[-1].split(' value="')[1].split('" ')[0]    
    return result


def imgname2result(imgname, taskxml, resultxml):
    key, task_value = taskxml2key(taskxml)
    img_label = key2result(resultxml, key)
    return imgname + ',' + task_value + ',' + img_label



def batch_repicture(args):
    sqls = glob.glob(args.spy_zip_path + '/*.sql')
    # sqls = [sys.path[0] + '/0724/pictures/6.sql']
    fw = open(args.spy_zip_path + '.txt', 'w')

    logging.info("sqls file size: %s" %(len(sqls)))
    logging.info("labelmap save path: %s" %(args.spy_zip_path + '.txt'))

    xml_root = os.path.join(args.spy_zip_path, "taskxml/")

    for item_sql in sqls:
        sql_inserts = readTxt(item_sql)

        logging.info("----------> Deal with %s, file size: %s" %(item_sql, len(sql_inserts)))

        for sql_insert in sql_inserts:
            if item_sql == os.path.join(args.spy_zip_path, '.sql'):
                logging.info(sql_insert)

            imgname, taskxml, resultxml = asql2img_chip_data(sql_insert)
            taskxml = xml_root + taskxml
            resultxml = xml_root + resultxml

            if not os.path.exists(taskxml):
                logging.info(imgname+" taskxml file not exists: "+taskxml)
                continue
            if not os.path.exists(resultxml):
                logging.info(imgname+" resultxml file not exists: "+resultxml)
                continue                
            
            ans = imgname2result(imgname, taskxml, resultxml)
            fw.write(ans + '\n')
    fw.close()

    logging.info("==================== Re CRCB shred Data Done ====================")

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--spy_zip_path', 
                        default=cfg.CRCB_DATA.SPY_ZIP_DIR, help='crcb org data')

    return parser.parse_args()

if __name__ == '__main__':
    batch_repicture(parse_args())
