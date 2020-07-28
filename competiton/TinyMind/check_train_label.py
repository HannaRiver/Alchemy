import os

RMBmap = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Q', 'W', 'E', 'R', 'A', 'H', 'S', 'X', 'J', 'Y', 'T', 'O', 'B', 'U', 'P', 'Z', 'C', 'L', 'F', 'G', 'D', 'M', 'I', 'K', 'N']

def checkInfo(txtpath):
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            infos = line.strip().split(' ')
            if len(infos) != 16:
                print("[Len Error]"+line + " not ok!")
                continue
            img_name = infos[0]
            labels = infos[1:]
            if len(labels) != 15:
                print(str(labels) + ' != 15')
                continue
            gt_label = os.path.basename(img_name).split('_')[0]
            if len(gt_label) != 10:
                print(gt_label + ' != 10')
                continue
            tag_label = get_ctc_decoder(labels, RMBmap)
            if len(tag_label) != 10:
                print(gt_label, tag_label, len(tag_label))
                continue
            if gt_label != tag_label:
                print(gt_label, tag_label)
                continue

            

def get_ctc_decoder(alist, maplist, black_id=-1):
    assert(len(maplist) == 36), "maplist error."
    if black_id == -1:
        black_id = maplist.index(' ')
    rlist = [int(alist[0])]
    for i in range(len(alist)-1):
        rlist.append(alist[i+1])
    rrlist = []
    for i in rlist:
        if int(i) == black_id:
            continue
        rrlist.append(i)
    return str('').join(maplist[int(i)] for i in rrlist)

if __name__ == '__main__':
    txtpath = '/work/competitions/TinyMind/net/ResNet30Rec/model/train.txt'
    checkInfo(txtpath)
    