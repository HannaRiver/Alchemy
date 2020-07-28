import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os

# img = cv2.imread('/home/hena/下载/2.png')
# img2 = cv2.imread('/home/hena/下载/3.png')
# cv2.rectangle(img, (0, 0), (1279, 64), (0, 0, 255), 1)
# cv2.rectangle(img, (0, 64), (640, 127), (0, 0, 255), 1)
# cv2.rectangle(img, (640, 65), (1279, 127), (0, 255, 0), 1)
# temp_img2 = img2[64: , : ]
# img3 = np.concatenate([img, temp_img2], axis=0)
# cv2.rectangle(img3, (0, 128), (640, 191), (0, 0, 255), 1)
# cv2.rectangle(img3, (640, 129), (1279, 191), (255, 0, 0), 1)
# cv2.imwrite('temp2.png', img3)
# print("Hi")

def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist

# img_list = readTxt('/work/tool/alchemy/temp.txt')
# img0 = np.ones([128, 128, 3], np.uint8)*255
# for i, item in enumerate(img_list):
#     img = cv2.imread(item)
#     color = (0, 0, 255) if i < 10 else (0, 255, 0)
#     cv2.rectangle(img, (0, 0), (127, 127), color, 1)
#     img0 = np.concatenate([img0, img], axis=1)

# cv2.imwrite('ygnet-1.png', img0)

# img0 = np.ones([32, 2688, 3], np.uint8)*255
# img1 = cv2.imread('/work/tool/alchemy/gt-1.png')
# img0 = np.concatenate([img0, img1], axis=0)
# cv2.imwrite('gt-1-.png', img0)
# img2 = cv2.imread('/work/tool/alchemy/KL-SP-1.png')
# img3 = cv2.imread('/work/tool/alchemy/rover-1.png')
# img4 = cv2.imread('/work/tool/alchemy/ygnet-1.png')

# 输入10帧预测20帧 上面一行是gt 下面是pre 红色表示的是gt pre表示预测

def concateImg(img_dir, other_dir, save_dir, img_size=480):
    img_name = os.path.basename(img_dir)
    gt_color_dir = os.path.join(img_dir, 'gt_color_')
    pred_color_dir = os.path.join(img_dir, 'pred_color_')
    colors = [[0, 0, 255], [0, 255, 0]] # gt, pred
    font=cv2.FONT_HERSHEY_SIMPLEX

    if not os.path.exists(pred_color_dir): return

    filenames = os.listdir(pred_color_dir)
    filenames.sort(key=lambda x:int(x[: -4]))

    gt_concat = np.ones([img_size+4, img_size+4, 3], np.uint8)*255 # 128+2+2
    cv2.putText(gt_concat, 'gt',(50,50), font, 1, colors[0], 2)

    gt_concat_input = np.ones([img_size+4, img_size+4, 3], np.uint8)*255
    gt_padding_input = np.zeros([img_size+4, img_size+4, 3], np.uint8)
    cv2.putText(gt_concat_input, 'input',(50,50), font, 1, colors[0], 2)

    pre_concat = np.ones([img_size+4, img_size+4, 3], np.uint8)*255
    cv2.putText(pre_concat, 'hq_pred',(0,50), font, 1, colors[1], 2)

    pre_concat_and = np.ones([img_size+4, img_size+4, 3], np.uint8)*255
    cv2.putText(pre_concat_and, 'taizhou_pred',(0,50), font, 1, colors[1], 2)

    pre_concat_and2 = np.ones([img_size+4, img_size+4, 3], np.uint8)*255
    cv2.putText(pre_concat_and2, 'mdata_pred',(0,50), font, 1, colors[1], 2)
    
    for i, filename in enumerate(filenames):
        if int(filename[: -4]) != i: print(img_dir)
        gt_path = os.path.join(gt_color_dir, filename)
        pre_path = os.path.join(pred_color_dir, filename)
        pre_path_and = os.path.join(other_dir, 'pred_color_', filename)
        # pre_path_and2 = os.path.join(other_dir2, 'pred_color_', filename)


        gt_img = cv2.imread(gt_path) if os.path.exists(gt_path) else np.zeros([img_size, img_size, 3], np.uint8)
        if i < 10:
            input_img = cv2.copyMakeBorder(gt_img,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[0])
            gt_concat_input = np.concatenate([gt_concat_input, input_img], axis=1)
            continue
        elif i < 20:
            gt_concat_input = np.concatenate([gt_concat_input, gt_padding_input], axis=1)
        pre_img = cv2.imread(pre_path)
        pre_img_and = cv2.imread(pre_path_and)
        # pre_img_and2 = cv2.imread(pre_path_and2) # 

        gt_img= cv2.copyMakeBorder(gt_img,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[0])
        pre_img= cv2.copyMakeBorder(pre_img,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[1])
        pre_img_and = cv2.copyMakeBorder(pre_img_and,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[1])
        # pre_img_and2 = cv2.copyMakeBorder(pre_img_and2,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[1])

        gt_concat = np.concatenate([gt_concat, gt_img], axis=1)
        pre_concat = np.concatenate([pre_concat, pre_img], axis=1)
        pre_concat_and = np.concatenate([pre_concat_and, pre_img_and], axis=1)
        # pre_concat_and2 = np.concatenate([pre_concat_and2, pre_img_and2], axis=1)
    concate_img = np.concatenate([gt_concat_input, gt_concat, pre_concat, pre_concat_and], axis=0)

    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, concate_img)
        

def concateImgs(img_root, other_root, save_dir, img_size=128):
    for item in os.listdir(img_root):
        save_path = os.path.join(save_dir, item)
        if os.path.exists(save_path):
            continue
        img_dir = os.path.join(img_root, item)
        other_dir = os.path.join(other_root, item)
        if not os.path.exists(other_dir):
            print(other_dir)
            continue

        # other_dir2 = os.path.join(other_root2, item)
        # if not os.path.exists(other_dir2):
        #     print(other_dir2)
        #     continue
        concateImg(img_dir, other_dir, save_dir, img_size)

def temp_concate(img_dir, save_root='', img_size=128):
    colors = [[0, 0, 255], [0, 255, 0]] # gt, pred
    img_name = os.path.basename(img_dir)
    gt_color_dir = os.path.join(img_dir, 'gt_color_')
    pred_color_dir = os.path.join(img_dir, 'pred_color_')
    save_dir = os.path.join(save_root)

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    filenames = os.listdir(pred_color_dir)
    filenames.sort(key=lambda x:int(x[: -4]))
    for i, filename in enumerate(filenames):
        gt_path = os.path.join(gt_color_dir, filename)
        pre_path = os.path.join(pred_color_dir, filename)
        save_path = os.path.join(save_dir, img_name + '-' + str(i) + '.png')
        

        gt_img = cv2.imread(gt_path) if os.path.exists(gt_path) else np.zeros([img_size, img_size, 3], np.uint8)
        if i < 10:
            input_img = cv2.copyMakeBorder(gt_img,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[0])
            gt_concat_input = np.concatenate([input_img, input_img], axis=1)
            cv2.imwrite(save_path, gt_concat_input)
            continue

        # gt_img = cv2.imread(gt_path) if os.path.exists(gt_path) else np.zeros([img_size, img_size, 3], np.uint8)
        pre_img = cv2.imread(pre_path)
        

        gt_img= cv2.copyMakeBorder(gt_img,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[0])
        pre_img= cv2.copyMakeBorder(pre_img,2,2,2,2,cv2.BORDER_CONSTANT,value=colors[1])
        gt_concat_pre = np.concatenate([gt_img, pre_img], axis=1)
        cv2.imwrite(save_path, gt_concat_pre)

        


def main():
    img_root = '/work/data/meteorology/radar/JiangSu/taizhou/0708_taizhou_400'
    other_root = '/work/data/meteorology/radar/JiangSu/taizhou/0708_taizhou_610testset'
    # other_root2 = '/work/data/meteorology/radar/JiangSu/taizhou/0708_taizhou_610testset'
    save_dir = '/work/data/meteorology/radar/JiangSu/taizhou/compose'
    # concateImgs(img_root, other_root, save_dir, 128)
    img_dir = '/work/data/meteorology/radar/JiangSu/taizhou/0708_taizhou_610testset'
    for item in os.listdir(img_dir):
        img_path = os.path.join(img_dir, item)
        temp_concate(img_path, '/work/data/meteorology/radar/JiangSu/taizhou/compose')

if __name__ == '__main__':
    main()
    
