import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET

def change2roi(xmlpath, xmlsave_path, fin_x1, fin_y1, fin_x3, fin_y3, padding_size):
    if not os.path.exists(xmlpath):
        print("%s doesn't exist." , xmlpath)
        return  
    xmlfile = open(xmlpath)
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    node_size = tree.find('size')
    for item in node_size:
        if item.tag=='width':
            item.text = str(fin_x3-fin_x1)
        if item.tag=='height':
            item.text = str(fin_y3-fin_y1)

    nodes = tree.findall('object')
    
    for node in nodes:
        for nod in node:
            if nod.tag=='bndbox':
                x1_o = max(0,int(nod[0].text)+padding_size-fin_x1)
                y1_o = max(0,int(nod[1].text)+padding_size-fin_y1)
                x2_o = min(fin_x3-fin_x1+1,int(nod[2].text)+padding_size-fin_x1)
                y2_o = max(0,int(nod[3].text)+padding_size-fin_y1)
                x3_o = min(fin_x3-fin_x1+1,int(nod[4].text)+padding_size-fin_x1)
                y3_o = min(fin_y3-fin_y1+1,int(nod[5].text)+padding_size-fin_y1)
                x4_o = max(0,int(nod[6].text)+padding_size-fin_x1)
                y4_o = min(fin_y3-fin_y1+1,int(nod[7].text)+padding_size-fin_y1)
                for i in range(len(nod)-1,-1,-1):
                    nod.remove(nod[i])

                x1 = ET.SubElement(nod, 'x1')
                x1.text = str(x1_o)
                y1 = ET.SubElement(nod, 'y1')
                y1.text = str(y1_o)
                x2 = ET.SubElement(nod, 'x2')
                x2.text = str(x2_o)
                y2 = ET.SubElement(nod, 'y2')
                y2.text = str(y2_o)
                x3 = ET.SubElement(nod, 'x3')
                x3.text = str(x3_o)
                y3 = ET.SubElement(nod, 'y3')
                y3.text = str(y3_o)
                x4 = ET.SubElement(nod, 'x4')
                x4.text = str(x4_o)
                y4 = ET.SubElement(nod, 'y4')
                y4.text = str(y4_o)

    tree.write(xmlsave_path)

def random_crop(img_path, xml_path, name, imgsave_dir, xmlsave_dir, padding_size, num_ofnew):
    img_ori = cv2.imread(img_path)
    for i in range(0,num_ofnew):
        imgsave_path = os.path.join(imgsave_dir,name+'_'+str(i)+'.jpg')
        xmlsave_path = os.path.join(xmlsave_dir,name+'_'+str(i)+'.xml')

        h_o,w_o,c_o = img_ori.shape
        img = cv2.copyMakeBorder(img_ori,padding_size,padding_size,padding_size,padding_size,cv2.BORDER_REPLICATE)

        h_offset1 = np.random.randint(0, padding_size, size=1)[0]
        w_offset1 = np.random.randint(0, padding_size, size=1)[0]
        h_offset2 = np.random.randint(0, padding_size, size=1)[0]
        w_offset2 = np.random.randint(0, padding_size, size=1)[0]

        x1 = w_offset1
        y1 = h_offset1
        x3 = w_o+2*padding_size-w_offset2
        y3 = h_o+2*padding_size-h_offset2
        img_post = img[y1:y3, x1:x3]
        change2roi(xml_path, xmlsave_path, x1, y1, x3, y3, padding_size)
        cv2.imwrite(imgsave_path, img_post)


if __name__ == '__main__':
    root_dir = '/data_1/project/competition/VOC2007'
    aug_number = 10
    padding_size = 60
    img_dir = os.path.join(root_dir, 'JPEGImages')
    xml_dir = os.path.join(root_dir, 'Annotations')

    imgsave_dir = os.path.join(root_dir, 'img_aug')
    xmlsave_dir = os.path.join(root_dir, 'xml_aug')
    if not os.path.exists(imgsave_dir):
        os.mkdir(imgsave_dir)
    if not os.path.exists(xmlsave_dir):
        os.mkdir(xmlsave_dir)

    for item in os.listdir(img_dir):
        img_path = os.path.join(img_dir, item)
        res = item.split(".jpg")
        name = res[0]
        xml_path = os.path.join(xml_dir,name+'.xml')
        random_crop(img_path, xml_path, name, imgsave_dir, xmlsave_dir, padding_size, aug_number)