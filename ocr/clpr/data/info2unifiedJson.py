#/usr/bin/python3
#-*- encoding=utf-8 -*-

'''
适用范围： 违章项目
功能： 整合数据部的标注结果，生成统一的json格式文件(格式见./example.json)
逻辑： 按照是否有车牌信息分为clpr与clp，再根据是否是人工标注分modeled与普通
'''

import os
import sys
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..','..', '..', 'utils')))
import json
import shutil

from isPlateNum import isPlateNum
from json2roi import ploygon2rect
from file2list import readTxt


def plate_point2org_img_point(car_rect, plate_polygon):
    '''
    car_rect = [x, y, w, h]
    plate_point = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    --> 对应原图的[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    '''
    return [[int(car_rect[i]) + int(item[i]) for i in range(2)] for item in plate_polygon]

def plate_rect2org_img_rect(car_rect, plate_rect):
    x = int(plate_rect[0]) + int(car_rect[0])
    y = int(plate_rect[1]) + int(car_rect[1])
    w = int(plate_rect[2])
    h = int(plate_rect[3])

    return x, y, w, h

############################### 数据部标注数据 ###########################################

def VRegInfo2Json_clp_model(image_name, model_det_dir, artif_det_dir, car_det_dir, image_dir, save_json_dir, save_image_dir):
    '''
    能定位但是不能识别的数据（包括人工标注及模型标注）
    version = v1.0.2.0.0(车辆检测用的是segnet, 车牌定位用的是v0.0.2版)
    car_det_dir: 车辆定位txt路径; save_json_dir: 保存json文件的根目录;
    rec_path: 识别结果图片路径; model_det_dir: model车牌定位txt目录(txt保存格式为x y w h);
    artif_det_dir: 人工标注的json根目录
    image_dir: 图片根目录;
    car_det_dir: 车辆定位txt路径; save_json_dir: 保存json文件的根目录;
    '''
    info_dict, clpr_info, car_det = {}, {}, {}
    json_dict = {'version': "v1.0.2.0.0"}

    json_dict['image_name'] = image_name

    # ================================ 车辆定位部分开始写入 ===============================
    det_car_txt_path = os.path.join(car_det_dir, image_name[: -4] + '.txt')
    
    if not os.path.exists(det_car_txt_path):
        logging.info("兄弟，车辆检测的坐标txt文档路径你写错了，我帮你把整个code关掉了!(如果路径没错，就是文件缺失，可能为%s)" %(det_car_txt_path))
        sys.exit()
    
    rect = readTxt(det_car_txt_path)[0].split(' ')
    if len(rect) != 4:
        logging.info("车辆定位结果异常(%s), path: %s" %(str(rect), det_car_txt_path))
        return
    car_det['car_obj'] = {'rect': rect}

    # ================================ 车牌定位部分开始写入 ===============================
    det_json_path = os.path.join(artif_det_dir, image_name + '.json')
    det_txt_path = os.path.join(model_det_dir, image_name[: -4] + '.txt')

    if not os.path.exists(det_json_path):
        if not os.path.exists(det_txt_path):
            logging.info("Error! Image(%s) 的人工和模型标注的结果都不存在！！！" %(image_name))
            return
        # 能走到这里，说明这个有定位没识别的图片 之前model work的很好，所以是rect作为保存
        clpr_rect = readTxt(det_txt_path)[0].split(' ')
        if len(clpr_rect) != 4:
            logging.info("模型标注的坐标点数目不对:%s" %(clpr_rect))
            return
        
        clpr_info['plate_obj'] = {'rect': plate_rect2org_img_rect(rect, clpr_rect), 'score': 1}
    
        info_dict['car_det'] = car_det
        info_dict['clpr'] = clpr_info
    
        json_dict['objects'] = [info_dict]
    
        fw = open(os.path.join(save_json_dir, image_name + '.json'), 'w')
        json.dump(json_dict, fw, ensure_ascii=False)
    
        fw.close()
    
        shutil.copy(os.path.join(image_dir, image_name), os.path.join(save_image_dir, image_name))
        return
    
    # 下面是人工标注的部分
    with open(det_json_path, 'r') as json_f:
        data = json.load(json_f)
        obj = data['objects']
        for item in obj:
            # point = [int(i) for i in item['polygon']]
            point = item['polygon']
            if len(point) != 4:
                logging.info("人工标注的坐标点数目不对: %s" %(det_json_path))
                return
            point1, point2, point3, point4 = plate_point2org_img_point(rect, point)
            label = item['label']

            if label == 'danweihuikuan':
                clpr_info['plate_obj'] = {'rect': ploygon2rect([point1, point2, point3, point4]), 'point': point1 + point2 + point3 + point4, 'score': 1}
            elif label == 'gerenhuikuanjiwuka':
                clpr_info['spray_plate_obj'] = {'rect': ploygon2rect([point1, point2, point3, point4]), 'point': point1 + point2 + point3 + point4, 'score': 1}
            else:
                logging.info("Warring: 人工定位标签异常! --> %s, json path: %s" %(label, det_json_path))
                return

    info_dict['car_det'] = car_det
    info_dict['clpr'] = clpr_info

    json_dict['objects'] = [info_dict]

    fw = open(os.path.join(save_json_dir, image_name + '.json'), 'w')
    json.dump(json_dict, fw, ensure_ascii=False)

    fw.close()

    shutil.copy(os.path.join(image_dir, image_name), os.path.join(save_image_dir, image_name))



def VRegInfo2Json_clpr_model(rec_path, model_det_dir, artif_det_dir, car_det_dir, image_dir, pics_dir, save_json_dir, save_image_dir):
    '''
    模型标注正确且有识别结果的数据（包括人工标注及模型标注）
    version = v1.0.2.0.0(车辆检测用的是segnet, 车牌定位用的是v0.0.2版)
    模型检出的结果，人工确认过
    rec_path: 识别结果图片路径; model_det_dir: model车牌定位txt目录(txt保存格式为x y w h);
    artif_det_dir: 人工标注的json根目录
    image_dir: 图片根目录;
    car_det_dir: 车辆定位txt路径; save_json_dir: 保存json文件的根目录;
    '''
    info_dict, clpr_info,  plate_rec, car_det = {}, {}, {}, {}
    json_dict = {'version': "v1.0.2.0.0"}

    img_basename = os.path.basename(rec_path)
    plate_num_gt = img_basename.split('_')[0]
    plate_rec['plate_num_gt'] = plate_num_gt
    clpr_info['plate_rec'] = plate_rec # 车牌识别部分完毕

    # assert(isPlateNum(plate_num_gt)), "Error1.0 :: 车牌标注不符合规范 " + plate_num_gt
    if not isPlateNum(plate_num_gt):
        logging.info("Error1.0 :: 车牌标注不符合规范(%s), rec img path: %s" %(plate_num_gt, rec_path))
        return

    image_name = img_basename[len(plate_num_gt)+1: ]
    json_dict['image_name'] = image_name

    pics_path = os.path.join(pics_dir, image_name)


    # ================================ 车辆定位部分开始写入 ===============================
    det_car_txt_path = os.path.join(car_det_dir, image_name[: -4] + '.txt')

    if not os.path.exists(det_car_txt_path):
        logging.info("兄弟，车辆检测的坐标txt文档路径你写错了，我帮你把整个code关掉了!(如果路径没错，就是文件缺失，可能为%s)" %(det_car_txt_path))
        sys.exit()
    
    rect = readTxt(det_car_txt_path)[0].split(' ')
    if len(rect) != 4:
        logging.info("车辆定位结果异常(%s), path: %s" %(str(rect), det_car_txt_path))
        return
    car_det['car_obj'] = {'rect': rect}

    # ================================ 车牌定位部分开始写入 ===============================
    det_txt_path = os.path.join(model_det_dir, image_name[: -4] + '.txt') # model result
    det_json_path = os.path.join(artif_det_dir, image_name + '.json') # Manually labeled results

    if not os.path.exists(det_txt_path):
        if not os.path.exists(det_json_path):
            logging.info("Error! Image(%s) 的人工和模型标注的结果都不存在！！！" %(det_txt_path))
            return
        # 能走到这里，说明这个图片是之前模型标注的不好的，人工有标签的，所以json出来的结果是point
        with open(det_json_path, 'r') as json_f:
            data = json.load(json_f)
            obj = data['objects']
            for item in obj:
                # point = [int(i) for i in item['polygon']]
                point = item['polygon']
                if len(point) != 4:
                    logging.info("人工标注的坐标点数目不对: %s" %(det_json_path))
                    return
                point1, point2, point3, point4 = plate_point2org_img_point(rect, point)
    
                label = item['label']

                # 这里的label需要你自己确认一下
                if label == 'danweihuikuan':
                    clpr_info['plate_obj'] = {'rect': ploygon2rect([point1, point2, point3, point4]), 'point': point1 + point2 + point3 + point4, 'score': 1}
                elif label == 'gerenhuikuanjiwuka':
                    clpr_info['spray_plate_obj'] = {'rect': ploygon2rect([point1, point2, point3, point4]), 'point': point1 + point2 + point3 + point4, 'score': 1}
                else:
                    logging.info("Warring: 人工定位标签异常! --> %s, json path: %s" %(label, det_json_path))
                    return
        info_dict['car_det'] = car_det
        info_dict['clpr'] = clpr_info        
        json_dict['objects'] = [info_dict]        
        fw = open(os.path.join(save_json_dir, image_name + '.json'), 'w')
        json.dump(json_dict, fw, ensure_ascii=False)        
        fw.close()        
        shutil.copy(os.path.join(image_dir, image_name), os.path.join(save_image_dir, image_name))
        os.remove(pics_path) # 删除pics是为了clp可以根据剩余的pics进行
        os.remove(det_json_path)

        return
    
    # 下面是model标注的很好的直接写入

    clpr_rect = readTxt(det_txt_path)[0].split(' ')

    if len(clpr_rect) != 4:
        logging.info("模型标注的坐标点数目不对:%s" %(clpr_rect))
        return
    
    clpr_info['plate_obj'] = {'rect': plate_rect2org_img_rect(rect, clpr_rect), 'score': 1}

    info_dict['car_det'] = car_det
    info_dict['clpr'] = clpr_info

    json_dict['objects'] = [info_dict]

    fw = open(os.path.join(save_json_dir, image_name + '.json'), 'w')
    json.dump(json_dict, fw, ensure_ascii=False)

    fw.close()

    shutil.copy(os.path.join(image_dir, image_name), os.path.join(save_image_dir, image_name))
    os.remove(pics_path)

############################### 车检数据 ###########################################
#### 这块数据还没有写解析代码 待写


###################################################################################

def main_clpr_modeled(v, keytime=0, det_version='v002',data_id='shuzhou_20180531'):
    '''
    这部分为经过人工核实确认模型标注正确的部分，由于目前都是定位矩形框，所以模型定位结果为rect（x, y, w, h）
    det_version: 车牌定位模型版本号（v002）
    keytime： 上传标注时候的日期（20181220）
    v： 批次号（5）
    '''
    logging.info("==================== Deal with clpr(%s) data ====================" %(data_id))

    org_data_dir = '/data_1/dataset/task2/batch' # 从违章林晓东那里拿的手持车牌数据目录
    save_dir = '/data_1/dataset/task2/shuzhou_20180531' # 标准化数据格式后保存的根目录

    logging.info("Data Id: %s \t Batch: %s \t UpdateKey: %s" %(data_id, v, keytime))
    logging.info("Plate Det Model Version: %s" %(det_version))
    logging.info("标准化数据格式后保存的根目录: %s" %(save_dir))

    model_det_dir = os.path.join(org_data_dir + v, det_version + '_txt')
    artif_det_dir = os.path.join(org_data_dir + v, 'json') # 人工标注定位的json路径
    car_det_dir = os.path.join(org_data_dir + v, 'txt_batch' + v)
    image_dir = os.path.join(org_data_dir + v, 'image_batch' + v)
    pics_dir = os.path.join(org_data_dir + v, 'pics') # 数据部门返回存在车牌的图片根目录
    save_json_dir = os.path.join(save_dir, 'batch' + v, 'clpr_json_' + det_version)
    save_image_dir = os.path.join(save_dir, 'batch' + v, 'clpr_img_' + det_version)

    save_json_dir_clp = os.path.join(save_dir, 'batch' + v, 'clp_json_' + det_version)
    save_image_dir_clp = os.path.join(save_dir, 'batch' + v, 'clp_img_' + det_version)

    rec_dir = os.path.join(org_data_dir + v, 'batch' + v + '_' + keytime)

    if not os.path.exists(os.path.join(save_dir, 'batch' + v)):
        os.mkdir(os.path.join(save_dir, 'batch' + v))

    if not os.path.exists(save_json_dir):
        os.mkdir(save_json_dir)

    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)

    if not os.path.exists(save_json_dir_clp):
        os.mkdir(save_json_dir_clp)

    if not os.path.exists(save_image_dir_clp):
        os.mkdir(save_image_dir_clp)

    # 根据人工标注的识别结果对数据进行分类
    if os.path.isdir(rec_dir):
        img_list = os.listdir(rec_dir)
    elif os.path.isfile(rec_dir):
        img_list = readTxt(rec_dir)

    logging.info("有车牌定位结果数据数量为: %s" %(len(os.listdir(pics_dir))))
    logging.info("------------------> Dear with clpr info >>>>>>>>>>>>>")
    all_cnt = len(img_list)
    logging.info("有车牌定位结果且定位清晰的数据数量为: %s" %(all_cnt))

    cnt_artif_beg = len(os.listdir(artif_det_dir))

    for rec_path in img_list:
        VRegInfo2Json_clpr_model(rec_path, model_det_dir, artif_det_dir, car_det_dir, image_dir, pics_dir, save_json_dir, save_image_dir)
    
    cnt_artif_end = len(os.listdir(artif_det_dir))
    logging.info("clpr-Artif-Label-Size: %s \t clpr-Model-Label-Size: %s" %(cnt_artif_beg-cnt_artif_end, all_cnt-cnt_artif_beg+cnt_artif_end))
    logging.info("=====================>>>>> clpr done! \n")

    logging.info("------------------> Dear with clp info >>>>>>>>>>>>>")
    all_cnt = len(os.listdir(pics_dir))
    logging.info("有车牌定位结果但无识别(车牌模糊人眼不能识别具体字符)的数据数量为: %s" %(all_cnt))
    logging.info("clp-Artif-Label-Size: %s \t clp-Model-Label-Size: %s" %(cnt_artif_end, all_cnt-cnt_artif_end))

    for image_name in os.listdir(pics_dir):
        VRegInfo2Json_clp_model(image_name, model_det_dir, artif_det_dir, car_det_dir, image_dir, save_json_dir_clp, save_image_dir_clp)
    
    logging.info("=====================>>>>> clp done! \n")
    
    

if __name__ == '__main__':
    v, keytime = sys.argv[1], sys.argv[2]
    main_clpr_modeled(v, keytime)

