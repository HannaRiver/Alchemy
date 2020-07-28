#/usr/bin/python3
#-*- coding=utf-8 -*-

import sys
import os
import logging
import random
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))
import cv2
import write_xml as wr_xml


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def get_bias_img_xml(img_path, xml_path, img_save_dir, xml_save_dir, bias_time=6):
    '''
    bias为向下截图的偏移量
    img: 原始图像
    xml: 原始图像对应的xml文件
    --> 保存新的截图和对应的xml文件
    '''
    img = cv2.imread(img_path)
    filename, w, h, d, name, bnbox = wr_xml.xml2info(xml_path)
    for i in range(bias_time):
        bn_xmin, bn_ymin, bn_xmax, bn_ymax = [int(i) for i in bnbox]
        if bn_ymin == 0:
            return
        bias = max(min(int((int(bn_ymin)) * random.random()), int(bn_ymin)-1), 1)
        
        img_name = str(bias) + '_' + filename

        img_save_path = os.path.join(img_save_dir, img_name)
        xml_save_path = os.path.join(xml_save_dir, img_name.split('.')[0]+'.xml')
        
        bn_ymin = bn_ymin - bias
        bn_ymax = bn_ymax - bias
        assert(bn_xmin > 0 and bn_xmax > 0 and bn_ymin > 0 and bn_ymax >0), str(bn_xmin)+' '+str(bn_ymin)+' '+str(bn_xmax)+' '+str(bn_ymax)+' '+'bnbox exceeds image boundary.'
        wr_xml.writeInfo2Xml(xml_save_path, img_name, name, bn_xmin, bn_ymin, bn_xmax, bn_ymax, w, h, d)
        
        bias_img = img[bias: int(h)+bias, :]
        cv2.imwrite(img_save_path, bias_img)

def batch_get_bias_img_xml(img_dir, xml_dir, img_save_dir, xml_save_dir):
    '''
    批量处理img_dir中的图像做偏移拓整和对应xml生成
    '''
    for adir in [img_save_dir, xml_save_dir]:
        if not os.path.exists(adir):
            os.mkdir(adir)
            logging.info('make %s folder.' %(adir))

    img_list = os.listdir(img_dir)
    cnt = 0
    for img_name in img_list:
        # img_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/title_whole_batch2'
        xml_path = os.path.join(xml_dir, img_name.split('.')[0]+'.xml')
        # img_name = img_name[len(img_name.split('_')[0])+1: ]
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            logging.info('None File :: %s' %(img_path))
            continue
        if not os.path.exists(xml_path):
            # logging.info('None File :: %s' %(xml_path))
            continue
        
        cnt += 1
        get_bias_img_xml(img_path, xml_path, img_save_dir, xml_save_dir)
    logging.info("File %s(%s) creat the bias img & xml done!" %(img_dir, cnt))

    


if __name__ == "__main__":
    # ==================================== batch1 ======================================
    # img_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/title_whole_batch1'
    # xml_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/batch1/xml'
    # img_save_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/batch1_aug_img'
    # xml_save_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/batch1_aug_xml'
    # batch_get_bias_img_xml(img_dir, xml_dir, img_save_dir, xml_save_dir)

    # ==================================== batch2 ======================================
    img_save_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/batch2_aug_img'
    xml_save_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/batch2_aug_xml'
    img_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/title_whole_batch2'
    xml_dir = '/work/hena/ocr/data/FinancialStatements/title/CRCB/batch2/model_tag/xml'

    batch_get_bias_img_xml(img_dir, xml_dir, img_save_dir, xml_save_dir)

    pass

