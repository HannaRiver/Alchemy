#/usr/bin/python3
#-*- coding=utf-8 -*-
import docx2txt
import os



def doc2docx(doc_path, docx_path):
    pass


def BatchDoc2Txt(doc_dir, txt_dir):
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    for item in os.listdir(doc_dir):
        doc_name = os.path.splitext(item)[0]
        doc_path = os.path.join(doc_dir, item)
        txt_path = os.path.join(txt_dir, doc_name + '.txt')

        text = docx2txt.process(doc_path)
        f = open(txt_path, 'w')
        f.write(text)

def Docx2Txt(docx_path, txt_path):
    f = open(txt_path, 'w')
    text = docx2txt.process(docx_path)
    f.write(text)

def main():
    docx_path = ''
    txt_path = ''
    Docx2Txt(docx_path, txt_path)

if __name__ == '__main__':
    main()
    