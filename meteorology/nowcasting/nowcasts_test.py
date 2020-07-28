import meteorology.nowcasting.emnowcasts as nowcasts
# import meteorology.nowcasting.nowcasting as nowcasts
import cv2
import numpy as np
import os

def readTxt(txtpath):
    imgs = np.zeros((10, 720, 720))
    with open(txtpath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            img = cv2.imread(line.strip(), 0)
            imgs[i] = img
    return imgs

def Test4c2r2c():
    img_lists = '/work/meteorology/nowcasting/data/error/20200107/test.txt'
    save_dir = '/work/meteorology/nowcasting/data/error/20200107/c2r2c'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    R = readTxt(img_lists)
    out = nowcasts.forecast(R)
    for i, item in enumerate(out):
        save_path = os.path.join(save_dir, str(i) + '.png')
        cv2.imwrite(save_path, nowcasts.dbz2colormap(item))
    
    return out

def tempfunc(img):
    # print(np.sum(img))
    img[img>0] = 1
    print(np.sum(img))
    return np.sum(img)

if __name__ == '__main__':
    # R = ''
    # R_f = c2r2c(R)
    Test4c2r2c()
