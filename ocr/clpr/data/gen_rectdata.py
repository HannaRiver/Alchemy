#-*- encoding=utf-8 -*-
import json
import os
import sys
from xml.dom.minidom import Document
from xml.etree.ElementTree import Element, ElementTree

import cv2
import math
import numpy as np
from shapely.geometry import *


def writeInfo2Xml(savename, img_name, target, w, h, d=3):
    doc = Document()

    elementlist = ['annotation', 'folder', 'filename', 'size', 'segmented'] # , 'object']
    detaillist = ['width', 'height', 'depth', 'name', 'pose', 'truncated', 'difficult', 'bndbox']
    bnbox_list = ['xmin', 'ymin', 'xmax', 'ymax']


    annotation, folder, filename, size, segmented= [doc.createElement(i) for i in elementlist]
    width, height, depth, name, pose, truncated, difficult, bndbox = [doc.createElement(i) for i in detaillist]
    xmin, ymin, xmax, ymax = [doc.createElement(i) for i in bnbox_list]

    doc.appendChild(annotation)

    folder_text, filename_text = [doc.createTextNode(i) for i in ['xml', img_name]]
        
    folder.appendChild(folder_text)
    filename.appendChild(filename_text)
    segmented.appendChild(doc.createTextNode('0'))

    width.appendChild(doc.createTextNode(str(w)))
    height.appendChild(doc.createTextNode(str(h)))
    depth.appendChild(doc.createTextNode(str(d)))
    for j in [width, height, depth]:
        size.appendChild(j)
    
    for j in [folder, filename, size, segmented]:
        annotation.appendChild(j)

    ###################### object ######################    
    for block in target:
        obj = doc.createElement('object')
        name, pose, truncated, difficult, bndbox = [doc.createElement(i) for i in detaillist[3:]]
        label_name = block[0]
        bn_xmin = block[1]
        bn_ymin = block[2]
        bn_xmax = block[1] + block[3]
        bn_ymax = block[2] + block[4]

        name.appendChild(doc.createTextNode(label_name))
        pose.appendChild(doc.createTextNode('Unspecified'))
        truncated.appendChild(doc.createTextNode('0'))
        difficult.appendChild(doc.createTextNode('0'))

    ################# object - bnbox #####################
        xmin, ymin, xmax, ymax = [doc.createElement(i) for i in bnbox_list]
        xmin.appendChild(doc.createTextNode(str(bn_xmin)))
        ymin.appendChild(doc.createTextNode(str(bn_ymin)))
        xmax.appendChild(doc.createTextNode(str(bn_xmax)))
        ymax.appendChild(doc.createTextNode(str(bn_ymax)))
        for j in [xmin, ymin, xmax, ymax]:
            bndbox.appendChild(j)
    
        for j in [name, pose, truncated, difficult, bndbox]:
            obj.appendChild(j)
        
        annotation.appendChild(obj)
    
    with open(savename, 'wb') as fw:
        fw.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    
    return

def txt2roi(txt_path, img_path, name, xmlsave_dir, marksave_dir):
    with open(txt_path, 'r') as txt_file:
        while True:
            lines = txt_file.readline()
            if not lines:
                break
                pass
            # print(lines)
            img_ori = cv2.imread(img_path)
            img_mark = img_ori.copy()
            h_o,w_o,c_o = img_ori.shape

            l = lines.split()
            r = [int(l[0]), int(l[1]), int(l[2]), int(l[3])]

            if r[0]<0 or r[1]<0 or r[0]+r[2]>w_o or r[1]+r[3]>h_o:
                print('Invalid point~')
                print(img_path)
                continue

            target = [['shouxiezi',r[0],r[1],r[0]+r[2],r[1]+r[3]]]
            marksave_path = os.path.join(marksave_dir, name+'.jpg')
            cv2.rectangle(img_mark, (r[0] , r[1]), (r[0]+r[2], r[1]+r[3]), (0,0,255), 2)
            # cv2.imshow('img_car',img_mark)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     return

            cv2.imwrite(marksave_path, img_mark)
            xml_savepath = os.path.join(xmlsave_dir,name+'.xml')
            writeInfo2Xml(xml_savepath, name+'.jpg', target, w_o, h_o, c_o)

def json2xml(json_path,img_path,name,xmlsave_dir,marksave_dir):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        objs = data['objects']

        classes = ['shouxiezi', 'jinzhangdan']
        img_ori = cv2.imread(img_path)
        h_o, w_o, c_o = img_ori.shape
        img_mark = img_ori.copy()
        target = []
        for obj in objs:
            label = obj['label']
            # print(label)
            if label not in classes:
                continue
            
            p = obj['polygon']
            x1, y1 = int(p[0][0]), int(p[0][1])
            x2, y2 = int(p[1][0]), int(p[1][1])
            x3, y3 = int(p[2][0]), int(p[2][1])
            x4, y4 = int(p[3][0]), int(p[3][1])
            a = [x1,y1,x2,y2,x3,y3,x4,y4]
            b = np.array(a).reshape(4,2)
            r = cv2.boundingRect(b)


            if r[0]<0 or r[1]<0 or r[0]+r[2]>w_o or r[1]+r[3]>h_o:
                print('Invalid point~')
                print(img_path)
                continue

            target.append([label,r[0],r[1],r[0]+r[2],r[1]+r[3]])
            
            color = (0,0,255) if label=='shouxiezi' else (0,255,255)
            cv2.rectangle(img_mark, (r[0] , r[1]), (r[0]+r[2], r[1]+r[3]), color, 2)

        # cv2.imshow('show', img_mark)
        # cv2.waitKey(0)

        xmlsave_path = os.path.join(xmlsave_dir, name+'.xml')
        writeInfo2Xml(xmlsave_path, name+'.jpg', target, w_o, h_o, c_o)
        marksave_path = os.path.join(marksave_dir, name+'.jpg')
        cv2.imwrite(marksave_path, img_mark)


def txt2xml():
    img_dir='/data_1/project/plates/data/suzhou_20180531/batch6/roi_batch6'
    txt_dir='/data_1/project/plates/data/suzhou_20180531/batch6/v002_txt'

    save_dir = '/data_1/project/plates/data/suzhou_20180531/batch6/test'
    # imgsave_dir=os.path.join(save_dir,'img')
    xmlsave_dir=os.path.join(save_dir,'xml')
    marksave_dir=os.path.join(save_dir,'mark')
    # if not os.path.exists(imgsave_dir):
    #     os.mkdir(imgsave_dir)
    if not os.path.exists(xmlsave_dir):
        os.mkdir(xmlsave_dir)
    if not os.path.exists(marksave_dir):
        os.mkdir(marksave_dir)

    for item in os.listdir(txt_dir):
        txt_path = os.path.join(txt_dir, item)
        res = item.split(".")
        name = res[0]
        img_path = os.path.join(img_dir,name+'.jpg')
        # print(img_path)
        txt2roi(txt_path,img_path,name,xmlsave_dir,marksave_dir)

def markjson2xml():
    img_dir='/data_1/project/plates/data/polygon/1/img'
    json_dir='/data_1/project/plates/data/polygon/1/njson'

    save_dir='/data_1/project/plates/data/polygon/1/test'
    # imgsave_dir=os.path.join(save_dir,'img')
    xmlsave_dir=os.path.join(save_dir,'xml')
    marksave_dir=os.path.join(save_dir,'mark')
    # if not os.path.exists(imgsave_dir):
    #     os.mkdir(imgsave_dir)
    if not os.path.exists(xmlsave_dir):
        os.mkdir(xmlsave_dir)
    if not os.path.exists(marksave_dir):
        os.mkdir(marksave_dir)

    for item in os.listdir(json_dir):
        json_path = os.path.join(json_dir, item)
        res = item.split(".")
        name = res[0]
        img_path = os.path.join(img_dir,name+'.jpg')
        # print(json_path)
        # print(img_path)
        json2xml(json_path,img_path,name,xmlsave_dir,marksave_dir)


if __name__ == '__main__':
    txt2xml()
