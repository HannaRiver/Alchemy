#-*- coding=utf-8 -*-
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import cv2
import shutil
import numpy as np
from PIL import Image, ImageDraw
from xml.etree.ElementTree import ElementTree, Element
# Make sure that caffe is on the python path:
# caffe_root = '/home/hena/caffe-ssd/caffe'
caffe_root = '/data_1/aws/textboxes++/TextBoxes_plusplus-master'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2
os.environ['GLOG_minloglevel'] = '2'


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file, mean_value=[185, 185, 185], scale=255):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize_h = image_resize[0]
        self.image_resize_w = image_resize[1]
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array(mean_value)) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', scale)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.2, topn=1):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result
    
    def detect_quad(self, image_file, conf_thresh=0.2, topn=1):
        '''
        TextBoxes++
        QUAD point
        '''
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        image = caffe.io.load_image(image_file)

        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']
        det_tmp = detections[0,0,:,0]
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
    	det_x1 = detections[0,0,:,7]
        det_y1 = detections[0,0,:,8]
        det_x2 = detections[0,0,:,9]
        det_y2 = detections[0,0,:,10]
        det_x3 = detections[0,0,:,11]
        det_y3 = detections[0,0,:,12]
        det_x4 = detections[0,0,:,13]
        det_y4 = detections[0,0,:,14]

        my_top_indice = [(i, conf) for i, conf in enumerate(det_conf) if conf>=conf_thresh and 1>det_x1[i]>0 and 1>det_y1[i]>0
         and 0<det_x2[i]<1 and 0<det_y2[i]<1 and 0<det_x3[i]<1 and 0<det_y3[i]<1 and 0<det_x4[i]<1 and 0<det_y4[i]<1]
        my_top_indice.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in my_top_indice]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        top_x1 = det_x1[top_indices]
        top_y1 = det_y1[top_indices]
        top_x2 = det_x2[top_indices]
        top_y2 = det_y2[top_indices]
        top_x3 = det_x3[top_indices]
        top_y3 = det_y3[top_indices]
        top_x4 = det_x4[top_indices]
        top_y4 = det_y4[top_indices]

        ################ hena #######################
        label_cls = list(set(top_label_indices))
        final_list = []
        for item in label_cls:
            lable_index = [i for i, cls_info in enumerate(top_label_indices) if cls_info==item]
            lable_index = lable_index[: topn]
            final_list += lable_index



        #############################################


        result = []
        # for i in xrange(min(topn, top_conf.shape[0])):
        for i in final_list:
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            result.append([xmin, ymin, xmax, ymax, label, score])
        return result


def save_ssd_result(args, image_dir, save_dir):
    '''将ssd定位的结果画出来并保存'''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        save_path = os.path.join(save_dir, img_name)
        get_bnbox(img_path, detection, save_path)

def get_bnbox(image_file, detection, save_path='', showkey=False):
    result = detection.detect(image_file)
    if len(result) == 0:
        shutil.copy(image_file, save_path)
        return False
    result = result[0]
    img = cv2.imread(image_file)
    height, width = img.shape[: 2]
    xmin, xmax = [int(round(result[i] * width)) for i in [0, 2]]
    ymin, ymax = [int(round(result[i] * height)) for i in [1, 3]]

    if xmin < 0 or ymin < 0 or ymax <= ymin or xmax <= xmin:
        cv2.imwrite(save_path, img)
        return False
    
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    if showkey:
        cv2.imshow("ssd result", img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(save_path, img)
    return True

def get_bnbox_quad(image_file, detection, save_path='', showkey=False):
    results = detection.detect_quad(image_file, 0.0, 1)
    if len(results) == 0:
        if not showkey:
            shutil.copy(image_file, save_path)
        return False
    result = results[0]
    img = cv2.imread(image_file)
    height, width = img.shape[: 2]
    for result in results:
        label = result[4]
        score = str(result[5])
        # print(label, score)
        xmin, xmax = [int(round(result[i] * width)) for i in [0, 2]]
        ymin, ymax = [int(round(result[i] * height)) for i in [1, 3]]

        if xmin < 0 or ymin < 0 or ymax <= ymin or xmax <= xmin:
            if not showkey:
                cv2.imwrite(save_path, img)
            return False
        color = (0, 0, 255) if label == 1 else (0, 255, 255)
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        if label == 1 :
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            font=cv2.FONT_HERSHEY_SIMPLEX
            sp = (10,40) if label == 1 else (10,80)
            cv2.putText(img, ("score:"+score), sp, font, 1, color, 2)
        
        if showkey and label == 1:
            cv2.namedWindow("ssd result", cv2.WINDOW_NORMAL)
            cv2.imshow("ssd result", img)
            cv2.waitKey(0)
        else:
            # pass
            cv2.imwrite(save_path, img)
    return True

def get_bnbox_preproess(image_file, detection, txtpath='', save_path='', save_roi_path='', plate_type_dict=None, showkey=False):
    results = detection.detect_quad(image_file, 0.2, 1)
    if len(results) == 0:
        shutil.copy(image_file, save_path)
        return False
    txt_file = open(txtpath, 'w')
    img = cv2.imread(image_file)
    result = results[0]
    height, width = img.shape[: 2]
    for result in results:
        label = result[4]
        
        xmin, xmax = [int(round(result[i] * width)) for i in [0, 2]]
        ymin, ymax = [int(round(result[i] * height)) for i in [1, 3]]

        if xmin < 0 or ymin < 0 or ymax <= ymin or xmax <= xmin:
            if not showkey:
                cv2.imwrite(save_path, img)
            return False

        color = (0, 0, 255) if label == 1 else (255, 0, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        if label == 1 or label == 2:
            roi_img = img[ymin: ymax, xmin: xmax]
            roi_path = save_roi_path +"_"+ plate_type_dict[str(label)]+'.jpg'
            cv2.imwrite(roi_path, roi_img)
        
        txt_file.write(str(label)+" "+str(xmin)+" "+str(ymin)+" "+str(xmax-xmin)+" "+str(ymax-ymin)+"\n")

    if showkey:
        cv2.namedWindow("ssd result", cv2.WINDOW_NORMAL)
        cv2.imshow("ssd result", img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(save_path, img)
    
    return True

def get_single_result(image_file, detection, check_dir):
    result = detection.detect(image_file)
    # print result

    img = Image.open(image_file)
    # draw = ImageDraw.Draw(img)
    img_name = image_file.split('/')[-1]
    actural_label = img_name.split('_')[0]
    width, height = img.size
    # print width, height
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        # draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        # draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
        if str(item[-1]) == str(actural_label) and xmin + xmax > 1:
            return 1
    # img.save(check_dir + img_name)
    return 0
        # print item
        # print [xmin, ymin, xmax, ymax]
        # print [xmin, ymin], item[-1]
    # img.save(check_dir + img_name)   

def nonlabel_result(image_file, detection, check_dir):
    '''
    完全没有标签，只要有框出来就认为对
    '''
    result = detection.detect(image_file)
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    img_name = image_file.split('/')[-1]
    width, height = img.size
    if len(result) == 0:
        print "None object!"
        img.save(check_dir + img_name)
        return 0
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
        if xmin + xmax > 1:
            img.save(check_dir + str(item[-1]) + '_' + img_name)
            return 1
    img.save(check_dir + 'bn_' + img_name)
    return 0
        

def main(args, image_dir, check_dir):
    global file_cnts
    global right_cnts
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    file_cnt, right_cnt = 0, 0
    for image_file in [image_dir + i for i in os.listdir(image_dir)]:
        file_cnt += 1
        right_cnt += get_single_result(image_file, detection, check_dir)
    print "File cont :: ", file_cnt
    print "ACC :: ", right_cnt*1.0/file_cnt
    file_cnts = file_cnts + file_cnt
    right_cnts = right_cnts + right_cnt
    # result = detection.detect(image_file)
    # print result

    # img = Image.open(image_file)
    # draw = ImageDraw.Draw(img)
    # img_name = image_file.split('/')[-1]
    # width, height = img.size
    # print width, height
    # for item in result:
    #     xmin = int(round(item[0] * width))
    #     ymin = int(round(item[1] * height))
    #     xmax = int(round(item[2] * width))
    #     ymax = int(round(item[3] * height))
    #     draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
    #     draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
    #     print item
    #     print [xmin, ymin, xmax, ymax]
    #     print [xmin, ymin], item[-1]
    # img.save(check_dir + img_name)

def xml2bnbox_affter(xml_dir):
    tree = ElementTree()
    tree.parse(xml_dir)
    nodes = tree.findall('object')
    bnboxlist = []
    label_name = []
    for node in nodes:
        name = node.find('name').text
        if '+' in name:
            index_x = name.index('+')
            name = name[: index_x] + '_' + name[index_x+1: ]
        bnboxlist.append([int(node.find('bndbox').find(i).text) for i in ['xmin', 'ymin', 'xmax', 'ymax']])
        label_name.append(name)
    if bnboxlist == [] or label_name == []:
        return 0, 0, 0, 0, 0
    xmin, ymin = [min(item[i] for item in bnboxlist) for i in [0, 1]]
    xmax, ymax = [max(item[i] for item in bnboxlist) for i in [2, 3]]
    return xmin, ymin, xmax, ymax, label_name

def cal_bnarea(bn):
    return (bn[2] - bn[0]) * (bn[3] - bn[1])

def iou(bn1, bn2):
    bn = [max(bn1[i], bn2[i]) for i in [0, 1]] + [min(bn1[i], bn2[i]) for i in [2, 3]]
    return cal_bnarea(bn)*1.0 / (cal_bnarea(bn1) + cal_bnarea(bn2) - cal_bnarea(bn))

def label_result(image_file, xml_file, check_dir, detection):
    '''有xml标签的测试code"'''
    # print "Deal with : ", image_file
    right_cnt, bnbox_cnt = 0, 0
    result = detection.detect(image_file)

    # tree = ElementTree()
    # tree.parse(xml_file)

    # nodes = tree.findall('object')
    # for node in nodes:
    #     actural_label = node.find('name').text
    x_min, y_min, x_max, y_max, actural_label = xml2bnbox_affter(xml_file)
    
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    img_name = image_file.split('/')[-1]
    draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0))
    if len(result) == 0:
        print "None object, File :: ", image_file
        img.save(check_dir + img_name)
        return 0, 0
    if len(result) > 1:
        print "Warring :: result not only! :: ", image_file
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
        if str(item[-1]) == str(actural_label[0]):
            right_cnt = 1
            if iou([x_min, y_min, x_max, y_max], [xmin, ymin, xmax, ymax]) > 0.7:
                bnbox_cnt = 1
            else:
                # draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0))
                img.save(check_dir + 'bn_' + img_name)
        else:
            if iou([x_min, y_min, x_max, y_max], [xmin, ymin, xmax, ymax]) > 0.7:
                bnbox_cnt = 1
                img.save(check_dir + str(item[-1]) + '_' + img_name)
            else:
                # draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0))
                img.save(check_dir + 'bn_' + str(item[-1]) + '_' + img_name)
        return right_cnt, bnbox_cnt

def label_nobnbox(image_file, xml_file, check_dir, detection):
    '''
    只比较分类结果，而不比较回归矩形框的结果
    '''
    result = detection.detect(image_file)
    x_min, y_min, x_max, y_max, actural_label = xml2bnbox_affter(xml_file)

    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    img_name = image_file.split('/')[-1]
    draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0))

    if len(result) == 0:
        return 0
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
        if str(item[-1]) == str(actural_label[0]):
            right_cnt = 1
            return 1
    return 0


def main_label(args, image_dir, xml_dir, check_dir):
    global file_cnts
    global right_cnts
    file_cnt, right_cnt, bnbox_cnt = 0, 0, 0
    detection = CaffeDetection(args.gpu_id, args.model_def, args.model_weights, args.image_resize, args.labelmap_file)
    for image_item in os.listdir(image_dir):
        image_file = image_dir + image_item
        xml_file = xml_dir + image_item[: -4] + '.xml'
        file_cnt += 1
        tmp_right, tmp_bnbox = label_result(image_file, xml_file, check_dir, detection)
        right_cnt += tmp_right
        bnbox_cnt += tmp_bnbox
    print "File cont :: ", file_cnt
    print "right_cnt :: %s, acc: %s" %(right_cnt, right_cnt * 1.0 / file_cnt)
    print "bnbox_cnt :: %s, acc: %s" %(bnbox_cnt, bnbox_cnt * 1.0 / file_cnt)
    file_cnts = file_cnts + file_cnt
    right_cnts = right_cnts + right_cnt

def main_label_nobnbox(args, image_dir, xml_dir, check_dir):
    '''只考虑label对不对，不考虑框对不对 '''
    global file_cnts
    global right_cnts
    file_cnt, right_cnt = 0, 0
    detection = CaffeDetection(args.gpu_id, args.model_def, args.model_weights, args.image_resize, args.labelmap_file)
    for image_item in os.listdir(image_dir):
        file_cnt += 1

        image_file = image_dir + image_item
        xml_file = xml_dir + image_item[: -4] + '.xml'

        tmp_right = label_nobnbox(image_file, xml_file, check_dir, detection)
        right_cnt += tmp_right

    print "File cont :: ", file_cnt
    print "right_cnt :: %s, acc: %s" %(right_cnt, right_cnt * 1.0 / file_cnt)      

    file_cnts = file_cnts + file_cnt
    right_cnts = right_cnts + right_cnt


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

def main_label_txt(args, image_txt, xml_txt, check_dir):
    file_cnt, right_cnt, bnbox_cnt = 0, 0, 0
    detection = CaffeDetection(args.gpu_id, args.model_def, args.model_weights, args.image_resize, args.labelmap_file)
    list_images = readTxt(image_txt)
    list_labels = readTxt(xml_txt)
    image_dir = '/work/hena/ocr/data/FinancialStatements/title/'
    for i, image_item in enumerate(list_images):
        image_file = image_dir + image_item
        xml_file = image_dir + list_labels[i]
        file_cnt += 1
        tmp_right, tmp_bnbox = label_result(image_file, xml_file, check_dir, detection)
        right_cnt += tmp_right
        bnbox_cnt += tmp_bnbox
    print "File cont :: ", file_cnt
    print "right_cnt :: %s, acc: %s" %(right_cnt, right_cnt * 1.0 / file_cnt)
    print "bnbox_cnt :: %s, acc: %s" %(bnbox_cnt, bnbox_cnt * 1.0 / file_cnt)

def tmp_save_result(args, img_dir):
    fw = open('/work/hena/ocr/data/FinancialStatements/title/CRCB/batch2/model_tag/info_list.txt', 'w')
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    check_list = readTxt('/work/hena/ocr/data/FinancialStatements/title/CRCB/batch2/model_tag/check_list.txt')
    for item in check_list:
        item_info = item.split('_')
        img_name = item_info[1] + '_' + item_info[2]
        image_file = img_dir + img_name
        label = item_info[0]
        result = detection.detect(image_file)
        if len(result) == 0:
            continue
        img = Image.open(image_file)
        width, height = img.size
        xmin = int(round(result[0][0] * width))
        ymin = int(round(result[0][1] * height))
        xmax = int(round(result[0][2] * width))
        ymax = int(round(result[0][3] * height))
        save_info = img_name + ',' + label + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n'
        fw.write(save_info)
    fw.close()



    
    

def main_nonlabel(args, image_dir, check_dir):
    file_cnt, right_cnt = 0, 0
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    for image_file in [image_dir + i for i in os.listdir(image_dir)]:
        file_cnt += 1
        right_cnt += nonlabel_result(image_file, detection, check_dir)
    print "File cont :: ", file_cnt
    print "ACC :: ", right_cnt*1.0/file_cnt

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/work/tmp/SignatureRec/model/CapitalMoneyRecog/labelmap.prototxt')
    parser.add_argument('--model_def',
                        default='/work/tmp/SignatureRec/model/CapitalMoneyRecog/CapitalMoneyRecogSSD.prototxt')
    parser.add_argument('--image_resize', default=[300, 300], type=int)
    parser.add_argument('--model_weights',
                        default='/work/tmp/SignatureRec/model/CapitalMoneyRecog/CapitalMoneyRecogSSD.caffemodel')
    return parser.parse_args()

file_cnts, right_cnts = 0, 0

if __name__ == '__main__':
    
    root_dir = "/work/hena/ocr/data/FinancialStatements/title/"
    
    # img_list = ['CIB/TEST/test_img/', 'CIB/TEST/test_JPEGImages/', 'CRCB/batch1/test_img/', 'CRCB/batch2/test_img/']
    # xml_list = ['CIB/TEST/test_xml/', 'CIB/TEST/test_Annotations/', 'CRCB/batch1/test_xml/', 'CRCB/batch2/test_xml/']
    
    # for index_i, img_dir in enumerate([root_dir + item for item in img_list]):
    #     xml_dir = root_dir + xml_list[index_i]
    #     check_dir = img_dir.split('_')[0] + '_check_v003/'
    #     if not os.path.exists(check_dir):
    #         os.mkdir(check_dir)
    #     print "========== " + img_list[index_i] + " =========="
    #     # main_label(parse_args(), img_dir, xml_dir, check_dir)
    #     main_label_nobnbox(parse_args(), img_dir, xml_dir, check_dir)
    # print "All File Cont :: ", file_cnts
    # print "Acc :: ", right_cnts * 1.0 / file_cnts
    # print "===================="
    
    image_txt = '/work/hena/ocr/model/caffe/title_ssd/v0.0.3/trainval_img.txt'
    xml_txt = '/work/hena/ocr/model/caffe/title_ssd/v0.0.3/trainvsave_pathal_label.txt'
    check_dir = '/work/hena/ocr/model/caffe/title_ssd/v0.0.3/tmp/'
    main_label_txt(parse_args(), image_txt, xml_txt, check_dir)

#__conda_setup="$(CONDA_REPORT_ERRORS=false '/home/jiangzhang/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    \eval "$__conda_setup"
#else
#    if [ -f "/home/jiangzhang/anaconda3/etc/profile.d/conda.sh" ]; then
#        . "/home/jiangzhang/anaconda3/etc/profile.d/conda.sh"
#        CONDA_CHANGEPS1=false conda activate base
#    else
#        \export PATH="/home/jiangzhang/anaconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup

