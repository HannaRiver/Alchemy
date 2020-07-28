#/usr/bin/python3
#-*- coding=utf-8 -*-

import sys
from xml.etree.ElementTree import ElementTree,Element
from xml.dom.minidom import Document


def writeInfo2Xml(savename, img_name, label_name, bn_xmin, bn_ymin, bn_xmax, bn_ymax, w, h, d=3):
    doc = Document()

    elementlist = ['annotation', 'folder', 'filename', 'size', 'segmented', 'object']
    detaillist = ['width', 'height', 'depth', 'name', 'pose', 'truncated', 'difficult', 'bndbox']
    bnbox_list = ['xmin', 'ymin', 'xmax', 'ymax']

    annotation, folder, filename, size, segmented, obj = [doc.createElement(i) for i in elementlist]
    width, height, depth, name, pose, truncated, difficult, bndbox = [doc.createElement(i) for i in detaillist]
    xmin, ymin, xmax, ymax = [doc.createElement(i) for i in bnbox_list]

    doc.appendChild(annotation)

    folder_text, filename_text = [doc.createTextNode(i) for i in ['xml', img_name]]
        
    folder.appendChild(folder_text)
    filename.appendChild(filename_text)
    segmented.appendChild(doc.createTextNode('0'))

    width.appendChild(doc.createTextNode(str(w)))
    height.appendChild(doc.createTextNode(str(h)))
    depth.appendChild(doc.createTextNode(str(d)))
    for j in [width, height, depth]:
        size.appendChild(j)
    
    ###################### object ######################
    name.appendChild(doc.createTextNode(label_name))
    pose.appendChild(doc.createTextNode('Unspecified'))
    truncated.appendChild(doc.createTextNode('0'))
    difficult.appendChild(doc.createTextNode('0'))

    ################# object - bnbox #####################
    xmin.appendChild(doc.createTextNode(str(bn_xmin)))
    ymin.appendChild(doc.createTextNode(str(bn_ymin)))
    xmax.appendChild(doc.createTextNode(str(bn_xmax)))
    ymax.appendChild(doc.createTextNode(str(bn_ymax)))
    for j in [xmin, ymin, xmax, ymax]:
        bndbox.appendChild(j)
    
    for j in [name, pose, truncated, difficult, bndbox]:
        obj.appendChild(j)
    
    for j in [folder, filename, size, segmented, obj]:
        annotation.appendChild(j)
    
    with open(savename, 'wb') as fw:
        fw.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    
    return

def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def xml2bnbox(xml_path):
    tree = read_xml(xml_path)
    nodes = tree.findall('object')
    for node in nodes:
        name = node.find('name').text
        bnbox = [int(node.find('bndbox').find(i).text) for i in ['xmin', 'ymin', 'xmax', 'ymax']]
    return name, bnbox

def xml2info(xml_path):
    tree = read_xml(xml_path)
    nodes = tree.findall('object')
    filename = tree.find('filename').text
    w, h, d = [tree.find('size').find(i).text for i in ['width', 'height', 'depth']]
    for node in nodes:
        name = node.find('name').text
        bnbox = [int(node.find('bndbox').find(i).text) for i in ['xmin', 'ymin', 'xmax', 'ymax']]
    return filename, w, h, d, name, bnbox

def modifyXml(xml_path, new_filename, save_xml_path):
    tree = read_xml(xml_path)
    node = tree.find('filename')
    node.text = new_filename

    tree.write(save_xml_path, encoding='utf-8')

def modifyXmlName(xml_path, new_labelname, save_xml_path):
    tree = read_xml(xml_path)
    nodes = tree.findall('object')
    for node in nodes:
        name = node.find('name')
        name.text = new_labelname
    tree.write(save_xml_path, encoding='utf-8')