import os


def readTxt(txtpath):
    fw = open('/work/competitions/TinyMind/data/finals/RBMCrownNumberRec236k_resize_check.txt', 'w')
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            line = os.path.basename(line.strip())
            label, img_name = line.split('_')[: 2]
            fw.write(img_name + ',' + label + '\n')
    fw.close()

if __name__ == '__main__':
    readTxt('/work/competitions/TinyMind/data/finals/result.txt')