#/usr/bin/python
#-*- encoding=utf-8 -*-
'''
Classification with LSTM caffe
Author: He Na
Time: 2018.12.29
'''
import os
import sys
from alchemy_config import cfg
utils_dir = cfg.UTILS_DIR
caffe_root = cfg.CAFFE_ROOT.LSTM
sys.path.insert(0, utils_dir)
import cv2
import numpy as np
import shutil
from file2list import readTxt
from resize_img import undeform_center_resize, resize_with_center_pad, undeform_resize
os.environ['GLOG_minloglevel'] = '2'
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe


class CaffeLstmClassification:
    def __init__(self, gpu_id, model_def, model_weights, time_step, image_resize, mean_value=[0, 0, 0], input_scale=1, resize_type=''):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize_h = image_resize[0]
        self.image_resize_w = image_resize[1]
        self.mean_value = mean_value
        self.resize_type = resize_type
        self.net = caffe.Net(model_def,
                             model_weights,
                             caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array(mean_value))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_input_scale('data', input_scale)
        self.transformer.set_channel_swap('data', (2, 1, 0))

        self.time_step = time_step
    
    def classify(self, image_file, layers='premuted_fc'):
        self.net.blobs['data'].reshape(1, 3, self.image_resize_h, self.image_resize_w)
        if self.resize_type == '':
            image = caffe.io.load_image(image_file) if type(image_file) == type('hello') else image_file / 255.0
        else:
            image = cv2.imread(image_file) if type(image_file) == type('Hi') else image_file      
            if self.resize_type == 'undeform_center_imgmean':
                mean_value = [np.mean(image[:,:,i]) for i in [0, 1, 2]]
                image = resize_with_center_pad(image, self.image_resize_h, self.image_resize_w, mean_value)/255.0
            elif self.resize_type == 'undeform_center':
                image = undeform_center_resize(image, self.image_resize_h, self.image_resize_w, self.mean_value)/255.0
            elif self.resize_type == 'undeform_resize':
                image = undeform_resize(image, self.image_resize_h, self.image_resize_w, self.mean_value)/255.0     
            elif self.resize_type == 'undeform_resize_white':
                image = undeform_resize(image, self.image_resize_h, self.image_resize_w, [255, 255, 255])/255.0
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image
        self.net.forward()

        prob = self.net.blobs[layers].data[0].flatten()
        prob = vector2array(prob, self.time_step)
        # print len(vector2array(prob))
        index_prob = [prob[i].argsort()[-1] for i in range(len(prob))]
        return prob, index_prob

def vector2array(avector, time_step=80):
    assert(len(avector)%time_step == 0), 'vector lenth error!'
    cls_num = len(avector)/time_step
    return [avector[cls_num*i: cls_num*(i+1)] for i in range(time_step)]

def get_ctc_decoder(alist, maplist, black_id=-1):
    if black_id == -1:
        black_id = maplist.index(' ')
    rlist = [alist[0]]
    for i in range(len(alist)-1):
        if alist[i+1] == alist[i]:
            continue
        rlist.append(alist[i+1])
    rrlist = []
    for i in rlist:
        if i == black_id:
            continue
        rrlist.append(i)
    return str('').join(maplist[i] for i in rrlist)


def main(image_list):
    '''
    这里只是举个例子怎么用，很少直接在当前代码下运行
    '''
    import argparse
    def parse_args():
        '''parse args'''
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_path', default='/work/hena/ocr/data/HandWriting/bill/CRCB/crcb_shred/cls_item/CA/CA/20180731/roi/0_大写金额_捌佰肆拾叁元整.png')
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
        parser.add_argument('--model_def',
                            default='/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/model/V0.1.1/ca_v011_deploy.prototxt')
        parser.add_argument('--image_resize', default=[48, 240], type=int)
        parser.add_argument('--time_step', default=60)
        parser.add_argument('--model_weights',
                            default='/work/hena/ocr/model/caffe/HandWriting/chn_lstm_ctc/chn_num/weights/V0.1.1/ca_iter_81000.caffemodel')
        return parser.parse_args()

    CAmap = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖',
             '拾', '元', '佰', '仟', '万', '亿', '角', '分', '整', '正',
             '圆', ' ']
    
    def stdCA(astr):
        if '圆' in astr:
            astr = astr.replace('圆', '元')
        if '正' in astr:
            astr = astr.replace('正', '')
    
        if '整' in astr:
            astr = astr.replace('整', '')
        return astr

    def CAmatch(label, prelabel):
        return stdCA(label) == stdCA(prelabel)

    args = parse_args()
    lstm_classification = CaffeLstmClassification(args.gpu_id,
                                                  args.model_def, args.model_weights,
                                                  args.time_step,
                                                  args.image_resize)
    save_prob = []
    cnt, right = 0, 0

    for i, image_path in enumerate(image_list):
        if not os.path.exists(image_path):
            continue
        cnt += 1
        label = image_path.split('_')[-1][: -4]
        dirname = os.path.dirname(image_path)
        
        prob, prob_index = lstm_classification.classify(image_path)
        prob_label = get_ctc_decoder(prob_index, CAmap, 21)
        if CAmatch(label, prob_label):
            right += 1
        else:
            print("[%s]%s %s" %(i, image_path, prob_label))
    print("ACC: ", right*1.0/cnt)

if __name__ == '__main__':
    img_list = readTxt('./train_labeled.txt')
    main(img_list)
