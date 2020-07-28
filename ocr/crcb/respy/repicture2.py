#!/usr/bin/env python
"""
repicture2.py is a map contact bin imgname, item.

Author: 8121

Time: 2018.9.12
"""

import os
import sys
import glob
from repicture import readTxt
from repicture import taskxml2key
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def sql2img_task(sql_insert):
    imgname, taskxml = [item.split('\\')[-1].split('||')[0] for item in sql_insert.split('${data')[1 :]]
    return imgname, taskxml

def imgname2task(imgname, taskxml):
    _, task_value = taskxml2key(taskxml)
    return imgname + ',' + task_value + ','


def batch_repicture2(spy_zip_path):
    sqls = glob.glob(spy_zip_path + '/*.sql')
    fw = open(spy_zip_path + '.txt', 'w')

    logging.info("sqls file size: %s" %(len(sqls)))
    logging.info("labelmap save path: %s" %(spy_zip_path + '.txt'))

    xml_root = os.path.join(spy_zip_path, "taskxml/")

    for item_sql in sqls:
        sql_inserts = readTxt(item_sql)

        logging.info("----------> Deal with %s, file size: %s" %(item_sql, len(sql_inserts)))

        for sql_insert in sql_inserts:
            if item_sql == os.path.join(spy_zip_path, '.sql'):
                logging.info(sql_insert)
            # print(sql_insert)
            imgname, taskxml = sql2img_task(sql_insert)
            taskxml = os.path.join(xml_root, taskxml)

            if not os.path.exists(taskxml):
                logging.info(imgname+" taskxml file not exists: "+taskxml)
                continue
            
            ans = imgname2task(imgname, taskxml)

            fw.write(ans + '\n')
    fw.close()

    logging.info("==================== Re CRCB shred Data Done ====================")

if __name__ == '__main__':
    spy_zip_path = '/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/spy/pictures'
    batch_repicture2(spy_zip_path)
