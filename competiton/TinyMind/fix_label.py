import os


def getInfo(txtpath):
    result = {}
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            infos = line.strip().split(',')
            if len(infos) != 2:
                print(line + " not ok!")
                continue
            img_name = infos[0]
            locate = infos[1]
            result[img_name] = locate
    return result

Dict_imgname_label = getInfo('/work/competitions/TinyMind/RBMCrownNumberRec1.txt')

def getInfo_fixed(txtpath):
    result = {}
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            img_name = os.path.basename(line.strip())
            img_id, label = img_name.split(',')
            if img_id in Dict_imgname_label:
                Dict_imgname_label[img_id] = label
    fw = open('/work/competitions/TinyMind/RBMCrownNumberRec3_new.csv', 'w')
    for item in Dict_imgname_label:
        fw.write(item + ',' + Dict_imgname_label[item] + '\n')
    fw.close()

if __name__ == '__main__':
    getInfo_fixed('/work/competitions/TinyMind/RBMCrownNumberRec3.txt')
    