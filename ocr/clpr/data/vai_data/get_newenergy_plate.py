import os
import shutil


def get_label4name(namestr, idx=0, split_key='_'):
    """
    通过解析文件名获取图像标签
    idx: 需要的内容在第几个项
    split_key: 分割字符串的key
    """
    if split_key not in namestr:
        return False
    return namestr.split(split_key)[idx]

def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
            # need_info = get_label4name(line.strip(), 0, ',')
            # if need_info:
            #     filelist.append(need_info)           
    return filelist

def get_newenergy_plate(img_dir, save_dir):
    for item in os.listdir(img_dir):
        if len(get_label4name(item, 1, '_')) == 7:
            img_path = os.path.join(img_dir, item)
            shutil.copy(img_path, os.path.join(save_dir, item))

if __name__ == '__main__':
    img_dir = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/suzhou/chepai_error'
    save_dir = '/work/hena/ocr/data/CLPR/VAI/NewEnergy/suzhou/pick_newenergy_error'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    get_newenergy_plate(img_dir, save_dir)

    
